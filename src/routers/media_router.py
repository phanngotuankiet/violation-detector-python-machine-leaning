from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import os
import cv2
from datetime import datetime
import shutil
from typing import List, Dict
from src.services.media_service import MediaService
import zipfile
from pathlib import Path
import tempfile
import numpy as np

router = APIRouter(prefix="/api/v1", tags=["media"])
media_service = MediaService(model_path="weights/crime_activity_v1.pt")

CONFIDENCE_THRESHOLD = 0.05


@router.post("/images/detect")
async def detect_image(image: UploadFile = File(...)):
    """Detect activities in a single image and return confidence scores for all labels"""
    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Please upload an image file."
        )

    try:
        # Read and process image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image_cv is None:
            raise ValueError("Could not read image file")

        # Resize image
        resized_image = cv2.resize(image_cv, (256, 256))

        # Get predictions
        results = media_service.model(resized_image)

        # Process all labels and their confidences
        label_confidences: Dict[str, float] = {}
        detected_labels: List[str] = []

        for r in results:
            probs = r.probs.data.cpu().numpy()
            for cls in range(len(probs)):
                conf = float(probs[cls])
                label = media_service.LABELS[cls]
                label_confidences[label] = conf

                # Check if confidence exceeds threshold for detection
                threshold = media_service.CRIME_THRESHOLDS.get(
                    label, CONFIDENCE_THRESHOLD
                )
                if conf > threshold and label != "normal videos":
                    detected_labels.append(label)

        return JSONResponse(
            {
                "status": "success",
                "data": {
                    "detected_activities": detected_labels,
                    "confidences": label_confidences,
                    "filename": image.filename,
                },
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/videos/process")
async def process_video(video: UploadFile = File(...)):
    """Process a single video file to detect criminal activities"""
    if not video.content_type.startswith("video/"):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Please upload a video file."
        )

    temp_video_path = (
        f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{video.filename}"
    )
    try:
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        result = await media_service.process_video(
            video_path=temp_video_path, conf_threshold=CONFIDENCE_THRESHOLD
        )

        return JSONResponse(
            {
                "status": "success",
                "data": {
                    "detected_crimes": result.detected_crimes,
                    "confidences": result.confidences,
                    "video_name": result.video_name,
                    "results_url": f"/api/v1/media/{result.video_name}/result",
                },
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)


@router.get("/media/{file_name}/result")
async def get_processed_results(file_name: str):
    """Retrieve all processed results as a ZIP file"""
    zip_path = None
    try:
        # Get all processed files
        result = media_service.get_processed_file_path(file_name)

        # Create temporary ZIP file
        zip_path = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file_name}.zip"

        # Create ZIP file
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            base_path_len = len(str(Path(result.base_path)))

            # Add all processed files to the ZIP
            for file_path in result.file_paths:
                # Calculate the relative path for the file in the ZIP
                arcname = str(Path(file_path))[base_path_len:].lstrip(os.sep)
                zipf.write(file_path, arcname)

        # Return the ZIP file
        return FileResponse(
            path=zip_path,
            media_type="application/zip",
            filename=f"{file_name}_results.zip",
        )

    except FileNotFoundError:
        if zip_path and os.path.exists(zip_path):
            os.remove(zip_path)
        raise HTTPException(status_code=404, detail="Processed files not found")
    except Exception as e:
        if zip_path and os.path.exists(zip_path):
            os.remove(zip_path)
        raise HTTPException(status_code=500, detail=str(e))
