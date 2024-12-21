from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO
import os
from fastapi import UploadFile


@dataclass
class VideoResult:
    detected_crimes: List[str]
    confidences: Dict[str, float]
    summary_path: str
    video_name: str
    output_dir: str


@dataclass
class ImageResult:
    detected_crimes: List[str]
    confidences: Dict[str, float]
    processed_path: str
    image_name: str
    output_dir: str


@dataclass
class ProcessedResult:
    base_path: str
    file_paths: List[str]


class MediaService:
    CRIME_THRESHOLDS = {
        "vandalism": 0.3,
        "shooting": 0.5,
        "explosion": 0.5,
        "arrest": 0.1,
        "assault": 0.05,
        "fighting": 0.05,
        "normal videos": 0.1,
        "road accidents": 0.1,
        "robbery": 0.1,
    }

    LABELS = [
        "arrest",
        "assault",
        "explosion",
        "fighting",
        "normal videos",
        "road accidents",
        "robbery",
        "shooting",
        "vandalism",
    ]

    def __init__(self, model_path: str, output_dir: str = "output"):
        self.model_path = model_path
        self.output_dir = output_dir
        self.model = self._initialize_model()
        os.makedirs(output_dir, exist_ok=True)

    def _initialize_model(self) -> YOLO:
        """Initialize the YOLO model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        model = YOLO(self.model_path).to("cuda")
        print(f"Model initialized on device: {model.device}")
        return model

    async def process_video(
        self, video_path: str, conf_threshold: float = 0.2
    ) -> VideoResult:
        """Process video and detect criminal activities"""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(self.output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)

        frames, predictions, detected_frames = self._process_video_frames(
            video_path, conf_threshold
        )

        if not frames:
            raise ValueError("Could not read video or video is empty")

        detected_crimes, crime_confidences = self._analyze_predictions(
            predictions, len(frames), conf_threshold
        )

        self._save_detected_frames(detected_frames, detected_crimes, video_output_dir)

        summary_path = self._generate_video_summary(
            frames, detected_crimes, crime_confidences, video_output_dir
        )

        return VideoResult(
            detected_crimes=detected_crimes,
            confidences=crime_confidences,
            summary_path=summary_path,
            video_name=video_name,
            output_dir=video_output_dir,
        )

    def _process_video_frames(
        self, video_path: str, conf_threshold: float
    ) -> Tuple[List[np.ndarray], List[Tuple[int, float]], Dict]:
        """Process individual frames from the video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        predictions = []
        detected_frames = {
            label: [] for label in self.LABELS if label != "normal videos"
        }
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frames.append(frame)
            resized_frame = cv2.resize(frame, (256, 256))
            results = self.model(resized_frame)

            self._process_frame_predictions(
                results,
                frame,
                frame_count,
                conf_threshold,
                predictions,
                detected_frames,
            )

            frame_count += 1

        cap.release()
        return frames, predictions, detected_frames

    def _process_frame_predictions(
        self,
        results,
        frame,
        frame_count: int,
        conf_threshold: float,
        predictions: list,
        detected_frames: dict,
    ):
        """Process predictions for a single frame with custom thresholds"""
        for r in results:
            probs = r.probs.data.cpu().numpy()
            for cls in range(len(probs)):
                conf = float(probs[cls])
                label = self.LABELS[cls]
                # Use custom threshold for each crime type
                threshold = self.CRIME_THRESHOLDS.get(label, conf_threshold)

                if conf > threshold and label != "normal videos":
                    predictions.append((cls, conf))
                    detected_frames[label].append(
                        {
                            "frame": frame.copy(),
                            "confidence": conf,
                            "frame_number": frame_count,
                        }
                    )

    def _analyze_predictions(
        self, predictions: list, total_frames: int, conf_threshold: float
    ) -> Tuple[List[str], Dict[str, float]]:
        """Analyze predictions to determine detected crimes"""
        crime_counts = {label: 0 for label in self.LABELS}
        for cls, _ in predictions:
            crime_counts[self.LABELS[cls]] += 1

        crime_ratios = {
            label: count / total_frames
            for label, count in crime_counts.items()
            if label != "normal videos"
        }

        detected_crimes = [
            label for label, ratio in crime_ratios.items() if ratio > conf_threshold
        ]

        crime_confidences = {}
        for crime in detected_crimes:
            crime_predictions = [
                p[1] for p in predictions if self.LABELS[p[0]] == crime
            ]
            if crime_predictions:
                crime_confidences[crime] = sum(crime_predictions) / len(
                    crime_predictions
                )

        return detected_crimes, crime_confidences

    def _save_detected_frames(
        self, detected_frames: dict, detected_crimes: list, output_dir: str
    ):
        """Save detected frames for each crime type"""
        for crime in detected_crimes:
            crime_dir = os.path.join(output_dir, crime)
            os.makedirs(crime_dir, exist_ok=True)

            sorted_frames = sorted(
                detected_frames[crime], key=lambda x: x["confidence"], reverse=True
            )

            for frame_data in sorted_frames:
                frame_path = os.path.join(
                    crime_dir,
                    f"frame_{frame_data['frame_number']}_"
                    f"conf_{frame_data['confidence']:.2f}.jpg",
                )
                cv2.imwrite(frame_path, frame_data["frame"])

    def _generate_video_summary(
        self,
        frames: List[np.ndarray],
        detected_crimes: List[str],
        crime_confidences: Dict[str, float],
        output_dir: str,
    ) -> str:
        """Generate and save summary frame for video"""
        middle_frame = frames[len(frames) // 2].copy()

        # Add text to summary frame
        text = "NORMAL"
        color = (0, 255, 0)

        if detected_crimes:
            crime_texts = [
                f"{crime} ({crime_confidences.get(crime, 0.0):.2f})"
                for crime in detected_crimes
            ]
            text = "CRIMES: " + ", ".join(crime_texts)
            color = (0, 0, 255)

        # Add text to frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), _ = cv2.getTextSize(text, font, 1, 2)

        cv2.rectangle(
            middle_frame,
            (10, 10),
            (20 + text_width, 20 + text_height),
            (255, 255, 255),
            -1,
        )

        cv2.putText(middle_frame, text, (15, 15 + text_height), font, 1, color, 2)

        # Save summary
        summary_path = os.path.join(output_dir, "summary.jpg")
        cv2.imwrite(summary_path, middle_frame)
        return summary_path

    async def process_image(
        self, image_file: UploadFile, conf_threshold: float = 0.1
    ) -> ImageResult:
        """Process a single image with custom thresholds"""
        contents = await image_file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not read image file")

        image_name = os.path.splitext(image_file.filename)[0]
        output_dir = os.path.join(self.output_dir, image_name)
        os.makedirs(output_dir, exist_ok=True)

        resized_image = cv2.resize(image, (256, 256))
        results = self.model(resized_image)

        detected_crimes = []
        confidences = {}

        for r in results:
            probs = r.probs.data.cpu().numpy()
            for cls in range(len(probs)):
                conf = float(probs[cls])
                label = self.LABELS[cls]
                # Use custom threshold for each crime type
                threshold = self.CRIME_THRESHOLDS.get(label, conf_threshold)

                if conf > threshold and label != "normal videos":
                    detected_crimes.append(label)
                    confidences[label] = conf

        processed_path = self._generate_processed_image(
            image, detected_crimes, confidences, output_dir
        )

        return ImageResult(
            detected_crimes=detected_crimes,
            confidences=confidences,
            processed_path=processed_path,
            image_name=image_name,
            output_dir=output_dir,
        )

    def get_processed_file_path(self, file_name: str) -> ProcessedResult:
        """Get all processed files for a given input file"""
        base_dir = os.path.join(self.output_dir, file_name)

        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"No processed files found for: {file_name}")

        all_files = []
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith((".jpg", ".jpeg", ".png")):
                    all_files.append(os.path.join(root, file))

        if not all_files:
            raise FileNotFoundError(f"No processed files found for: {file_name}")

        return ProcessedResult(base_path=base_dir, file_paths=all_files)

    def _generate_processed_image(
        self,
        image: np.ndarray,
        detected_crimes: List[str],
        confidences: Dict[str, float],
        output_dir: str,
    ) -> str:
        """Generate and save processed image with annotations"""
        processed_image = image.copy()

        # Add text to image
        text = "NORMAL"
        color = (0, 255, 0)

        if detected_crimes:
            crime_texts = [
                f"{crime} ({confidences.get(crime, 0.0):.2f})"
                for crime in detected_crimes
            ]
            text = "CRIMES: " + ", ".join(crime_texts)
            color = (0, 0, 255)

        # Add text to image
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), _ = cv2.getTextSize(text, font, 1, 2)

        cv2.rectangle(
            processed_image,
            (10, 10),
            (20 + text_width, 20 + text_height),
            (255, 255, 255),
            -1,
        )

        cv2.putText(processed_image, text, (15, 15 + text_height), font, 1, color, 2)

        # Save processed image
        processed_path = os.path.join(output_dir, "processed.jpg")
        cv2.imwrite(processed_path, processed_image)
        return processed_path

    def get_processed_file_path(self, file_name: str) -> ProcessedResult:
        """Get all processed files for a given input file"""
        base_dir = os.path.join(self.output_dir, file_name)

        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"No processed files found for: {file_name}")

        # Get all files in the directory and its subdirectories
        all_files = []
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith((".jpg", ".jpeg", ".png")):
                    all_files.append(os.path.join(root, file))

        if not all_files:
            raise FileNotFoundError(f"No processed files found for: {file_name}")

        return ProcessedResult(base_path=base_dir, file_paths=all_files)
