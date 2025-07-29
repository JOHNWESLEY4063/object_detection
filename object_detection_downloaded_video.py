import cv2
import time
import os
import tkinter as tk
from tkinter import filedialog
import subprocess
from yolov8.YOLOv8 import YOLOv8

# Initialize Tkinter (for file dialog)
root = tk.Tk()
root.withdraw()  # Hide root window

# Ask the user to upload a video file
print("📂 Please select a video file to analyze...")
video_path = filedialog.askopenfilename(
    title="Select a Video File",
    filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv;*.flv")]
)

if not video_path:
    print("❌ No video selected! Exiting...")
    exit()

if not os.path.exists(video_path):
    print(f"❌ Selected video file '{video_path}' not found! Exiting...")
    exit()

print(f"✅ Selected video: {video_path}")

# Configurations
START_TIME = 5  # Skip the first 5 seconds
MODEL_PATH = "C:\\Users\\peram\\object_detection\\yolov8-onnx-env\\Scripts\\yolov8m.onnx"
MAX_RETRIES = 10  # Maximum retries if no frame is received

# Function to check if FFmpeg is installed
def is_ffmpeg_installed():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

# Function to fix video codec issues using FFmpeg
def fix_video_codec(input_path):
    fixed_video_path = os.path.splitext(input_path)[0] + "_fixed.mp4"

    if not is_ffmpeg_installed():
        print("⚠️ FFmpeg is not installed. Using the original video file.")
        return input_path

    print("⚙️ Checking and fixing video codec if needed...")

    try:
        command = [
            "ffmpeg", "-i", input_path, "-c:v", "libx264", "-preset", "slow",
            "-crf", "18", "-c:a", "aac", "-b:a", "128k", fixed_video_path, "-y"
        ]
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print(f"✅ Fixed video saved at: {fixed_video_path}")
        return fixed_video_path
    except subprocess.CalledProcessError:
        print("⚠️ FFmpeg failed to fix the video. Using the original file.")
        return input_path

# Check and fix video codec if necessary
video_path = fix_video_codec(video_path)

# Load video source
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Failed to open video source! Check the file format.")
    exit()

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
start_time = time.time()

while cap.isOpened():
    if cv2.waitKey(1) == ord("q"):
        print("🚪 Exiting on user request.")
        break

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
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
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

# Release resources
cap.release()
cv2.destroyAllWindows()