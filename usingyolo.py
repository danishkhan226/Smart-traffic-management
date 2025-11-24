import cv2
import torch
import time
from ultralytics import YOLO

# ---------------------------
# Configuration
# ---------------------------
CAMERA_INDEX = 0
IMG_SIZE = 640
ALPHA_COUNT = 0.6  # EMA smoothing for counts

# Confidence thresholds
AMBULANCE_CONF = 0.5
VEHICLE_CONF = 0.4

# Traffic classes for general vehicles
VEHICLE_CLASSES = ['car', 'bus', 'truck', 'motorbike', 'bicycle']

# Traffic light timing
GREEN_MIN = 5
GREEN_MAX = 20

# ---------------------------
# Load models
# ---------------------------
ambulance_model = YOLO(r"C:\Users\danis\Downloads\project\best.pt")
vehicle_model = YOLO(r"C:\Users\danis\Downloads\project\yolov8n.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
ambulance_model.to(device)
vehicle_model.to(device)

# ---------------------------
# Camera setup
# ---------------------------
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

# ---------------------------
# Initialize variables
# ---------------------------
smoothed_ns = 0
smoothed_ew = 0
current_green = "NS"
green_start_time = time.time()
green_duration = GREEN_MIN

# ---------------------------
# Main loop
# ---------------------------
def main():
    global smoothed_ns, smoothed_ew, current_green, green_start_time, green_duration

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ns_count = 0
        ew_count = 0

        line_x = frame.shape[1] // 2
        cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (0, 0, 255), 2)

        # -------- Detect ambulance --------
        ambulance_results = ambulance_model(frame, imgsz=IMG_SIZE)[0]
        ambulance_boxes = []

        for r in ambulance_results.boxes:
            cls_id = int(r.cls[0])
            conf = float(r.conf[0])
            if conf < AMBULANCE_CONF:
                continue
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            ambulance_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red box
            cv2.putText(frame, f"AMBULANCE {conf:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            cx = (x1 + x2) // 2
            if cx < line_x:
                ns_count += 1
            else:
                ew_count += 1

        # -------- Detect other vehicles --------
        vehicle_results = vehicle_model(frame, imgsz=IMG_SIZE)[0]
        for r in vehicle_results.boxes:
            cls_id = int(r.cls[0])
            conf = float(r.conf[0])
            class_name = vehicle_model.names[cls_id]
            if conf < VEHICLE_CONF or class_name not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, r.xyxy[0])

            # Check overlap with ambulance boxes
            overlap = False
            for ax1, ay1, ax2, ay2 in ambulance_boxes:
                if not (x2 < ax1 or x1 > ax2 or y2 < ay1 or y1 > ay2):
                    overlap = True
                    break
            if overlap:
                continue

            # Draw rectangle for vehicle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow box
            cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

            cx = (x1 + x2) // 2
            if cx < line_x:
                ns_count += 1
            else:
                ew_count += 1

        # -------- Smooth counts --------
        smoothed_ns = ALPHA_COUNT * ns_count + (1 - ALPHA_COUNT) * smoothed_ns
        smoothed_ew = ALPHA_COUNT * ew_count + (1 - ALPHA_COUNT) * smoothed_ew

        # -------- Adaptive traffic duration --------
        ns_green_time = min(max(int(smoothed_ns), GREEN_MIN), GREEN_MAX)
        ew_green_time = min(max(int(smoothed_ew), GREEN_MIN), GREEN_MAX)

        elapsed = time.time() - green_start_time
        if elapsed >= green_duration:
            if current_green == "NS":
                current_green = "EW"
                green_duration = ew_green_time
            else:
                current_green = "NS"
                green_duration = ns_green_time
            green_start_time = time.time()
            elapsed = 0

        remaining_time = int(green_duration - elapsed)
        ns_color = (0, 255, 0) if current_green=="NS" else (0,0,255)
        ew_color = (0, 255, 0) if current_green=="EW" else (0,0,255)

        # -------- Draw traffic info --------
        cv2.putText(frame, f"NS Count: {int(smoothed_ns)}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.circle(frame, (150, 100), 30, ns_color, -1)
        cv2.putText(frame, f"{remaining_time}", (135, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

        cv2.putText(frame, f"EW Count: {int(smoothed_ew)}", (frame.shape[1]-200, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.circle(frame, (frame.shape[1]-150, 100), 30, ew_color, -1)
        cv2.putText(frame, f"{remaining_time}", (frame.shape[1]-165, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

        # -------- Show frame --------
        cv2.imshow("Traffic Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print(f"Using device: {device}")
    main()
