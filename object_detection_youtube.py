import cv2
import time
import os
from yolov8.YOLOv8 import YOLOv8

# Try importing YouTube capture only if needed
try:
    from cap_from_youtube import cap_from_youtube
except ImportError:
    cap_from_youtube = None
    print("⚠️ cap_from_youtube module not found. YouTube streaming won't work.")

# Configurations
USE_LOCAL_VIDEO = False  # Set to False to use the YouTube video
VIDEO_URL = "https://www.youtube.com/watch?v=MNn9qKG2UFI"  # Corrected YouTube URL
LOCAL_VIDEO_PATH = "video.mp4"  # Update this to the correct path if using a local file
START_TIME = 5  # Skip the first 5 seconds
MODEL_PATH = "C:\\Users\\peram\\object_detection\\yolov8-onnx-env\\Scripts\\yolov8m.onnx"

# Load video source
if USE_LOCAL_VIDEO:
    if not os.path.exists(LOCAL_VIDEO_PATH):
        print(f"❌ Local video '{LOCAL_VIDEO_PATH}' not found! Exiting...")
        exit()
    cap = cv2.VideoCapture(LOCAL_VIDEO_PATH)
else:
    if cap_from_youtube is None:
        print("❌ cap_from_youtube is not available. Use a local video instead.")
        exit()
    cap = cap_from_youtube(VIDEO_URL, resolution="720p")

# Check if video capture was successful
if not cap.isOpened():
    print("❌ Failed to open video source! Check your file path or internet connection.")
    exit()

print(f"✅ Video capture object created: {cap}")

# Set start time (use milliseconds for better accuracy)
cap.set(cv2.CAP_PROP_POS_MSEC, START_TIME * 1000)

# Load YOLOv8 model
if not os.path.exists(MODEL_PATH):
    print(f"❌ YOLOv8 model file '{MODEL_PATH}' not found! Exiting...")
    exit()
yolov8_detector = YOLOv8(MODEL_PATH, conf_thres=0.5, iou_thres=0.5)

# Create OpenCV window
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detected Objects", 800, 600)

frame_count = 0
retry_count = 0
MAX_RETRIES = 10  # Maximum retries if no frame is received
prev_time = time.time()

while cap.isOpened():
    if cv2.waitKey(1) == ord("q"):
        print("🚪 Exiting on user request.")
        break

    # Read frame with retry mechanism
    ret, frame = cap.read()
    if not ret:
        retry_count += 1
        print(f"⚠️ No frame received (attempt {retry_count}/{MAX_RETRIES})...")
        if retry_count >= MAX_RETRIES:
            print("❌ Max retries reached. Exiting...")
            break
        time.sleep(1)  # Wait before retrying
        continue  # Retry instead of breaking

    retry_count = 0  # Reset retry count on success
    frame_count += 1

    # Calculate FPS
    curr_time = time.time()
    fps = frame_count / (curr_time - prev_time)
    print(f"✅ Frame {frame_count} received | FPS: {fps:.2f}")

    # Save a test frame for debugging (only on the first frame)
    if frame_count == 1:
        cv2.imwrite("test_frame.jpg", frame)

    # Object Detection
    boxes, scores, class_ids = yolov8_detector(frame)

    # Draw detections on frame
    combined_img = yolov8_detector.draw_detections(frame)

    # Show output
    cv2.imshow("Detected Objects", combined_img)

cap.release()
cv2.destroyAllWindows()