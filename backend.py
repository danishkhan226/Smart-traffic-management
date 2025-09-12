from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from io import BytesIO
import uvicorn
import tempfile
import os
from ultralytics import YOLO
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8050"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model directly using YOLO class
try:
    yolo_model = YOLO('yolov8n.pt')  # 'n' for nano (fastest); use 's', 'm', 'l', 'x' for more accuracy
    VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck (COCO indices)
    print("Model loaded successfully")
except Exception as e:
    raise RuntimeError(f"Failed to load YOLO model: {e}. Ensure 'yolov8n.pt' is downloaded and compatible with ultralytics and PyTorch versions. Download from https://github.com/ultralytics/assets/releases if needed.")

def detect_vehicles_yolo(frame):
    # Run YOLO inference on the frame
    results = yolo_model(frame, verbose=False)
    vehicle_count = 0
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get class ID and confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if conf > 0.5 and cls in VEHICLE_CLASSES:  # Confidence threshold > 0.5, only vehicles
                    vehicle_count += 1
                    # Optional: Draw bounding box for visualization
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return vehicle_count

@app.post("/process_video/")
async def process_video(file: UploadFile = File(...)):
    try:
        # Read the uploaded file contents
        contents = await file.read()
        if not contents:
            raise ValueError("Empty video file uploaded")

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(contents)
            temp_file_path = temp_file.name

        # Open the temporary video file
        cap = cv2.VideoCapture(temp_file_path)
        if not cap.isOpened():
            os.unlink(temp_file_path)  # Clean up temporary file
            raise ValueError("Failed to open video file")

        total_vehicles = 0
        frame_count = 0
        vehicle_counts = []  # For detailed export

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            vehicles = detect_vehicles_yolo(frame)
            total_vehicles += vehicles
            vehicle_counts.append(vehicles)
            frame_count += 1

        cap.release()
        os.unlink(temp_file_path)  # Clean up temporary file

        if frame_count == 0:
            raise ValueError("No frames processed from video")
        average_vehicles = total_vehicles / frame_count
        congestion_level = "High" if average_vehicles > 10 else "Low"

        # Export to JSON
        with open("traffic_analysis.json", "w") as f:
            json.dump({
                "average_vehicles": average_vehicles,
                "congestion_level": congestion_level,
                "vehicle_counts": vehicle_counts
            }, f)

        return {
            "average_vehicles": average_vehicles,
            "congestion_level": congestion_level,
            "vehicle_counts": vehicle_counts
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    return {"status": "Backend is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)