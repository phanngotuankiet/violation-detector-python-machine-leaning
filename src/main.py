import sys

sys.path.append(".")

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.routers import media_router

# Create output directory if it doesn't exist
OUTPUT_DIR = "output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Initialize FastAPI app
app = FastAPI(
    title="Crime Detection API",
    description="API for detecting criminal activities in videos and images",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory for serving processed media
app.mount("/static", StaticFiles(directory="static", html=True), name="static")


# Add route for serving index.html
@app.get("/")
async def read_root():
    index_path = os.path.join("static", "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Index file not found")
    return FileResponse(index_path)


# Include routers
app.include_router(media_router.router)


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return {"status": "error", "message": str(exc), "path": request.url.path}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
