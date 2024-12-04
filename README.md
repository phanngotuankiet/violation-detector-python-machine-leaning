# Crime Detection API

A FastAPI-based service for detecting criminal activities in videos and images using computer vision and deep learning.

## Features

- Real-time crime detection in videos and images
- Support for multiple crime categories:
  - Vandalism
  - Shooting
  - Explosion 
  - Arrest
  - Assault
  - Fighting
  - Road accidents
  - Robbery
- Custom confidence thresholds for each crime type
- Processed media storage and retrieval
- ZIP archive generation for processed results
- REST API endpoints for video/image processing

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)
- Conda package manager
- Required Python packages:
```
fastapi==0.104.1
python-multipart==0.0.6
uvicorn==0.24.0
opencv-python==4.8.1.78
numpy==1.26.0
python-jose==3.3.0
python-dotenv==1.0.0
ultralytics
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <project-folder>
```

2. Create and activate a Conda environment:
```bash
# Create new environment
conda create -n crime-detection python=3.10
# Activate environment
conda activate crime-detection
```

3. Install CUDA dependencies (if using GPU):
```bash
# Install CUDA toolkit
conda install -c nvidia cuda-toolkit
```

4. Install dependencies:
```bash
# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other requirements
pip install -r requirements.txt
```

5. Place your trained YOLO model weights in the `weights` directory:
```
weights/crime_activity_v1.pt
```

## Usage

1. Activate the Conda environment:
```bash
conda activate crime-detection
```

2. Start the server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

3. API Endpoints:

- Process Video:
  ```http
  POST /api/v1/videos/process
  ```
  - Request: `multipart/form-data` with video file
  - Response: JSON with detected crimes, confidences, and results URL

- Get Processed Results:
  ```http
  GET /api/v1/media/{file_name}/result
  ```
  - Response: ZIP file containing processed media files and analysis results

## Output Structure

Processed results are stored in the `output` directory with the following structure:
```
output/
├── {video_name}/
│   ├── summary.jpg
│   ├── {crime_type}/
│   │   ├── frame_{number}_conf_{confidence}.jpg
│   │   └── ...
│   └── ...
```

## Error Handling

The API includes comprehensive error handling for:
- Invalid file types
- Processing failures
- File not found errors
- Server errors

All errors return appropriate HTTP status codes and detailed error messages.

## Development

- The application uses FastAPI for the REST API
- CORS is enabled for all origins
- Static files are served from the `output` directory
- Global exception handling is implemented

## Troubleshooting

Common CUDA-related issues:
1. If you encounter CUDA errors, verify your CUDA installation:
```bash
conda list cuda
nvidia-smi
```

2. If PyTorch isn't using GPU:
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

3. For environment conflicts:
```bash
# Create fresh environment
conda create -n crime-detection-new python=3.10 --no-defaults
conda activate crime-detection-new
# Follow installation steps again
```

## License

Ho Le Minh Hoang
0161001726415 - Vietcombank
Anh Hiển trả em 500k