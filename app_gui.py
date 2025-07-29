# app_gui.py

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import cv2
import time
import os
from yolov8.YOLOv8 import YOLOv8

# Conditional import for YouTube streaming
try:
    from cap_from_youtube import cap_from_youtube
except ImportError:
    cap_from_youtube = None

# --- Configuration ---
MODEL_PATH = "yolov8m.onnx"
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5

# --- Distance Estimation Constants (for Webcam) ---
FOCAL_LENGTH = 700  # <<-- CALIBRATE THIS VALUE
KNOWN_WIDTHS_CM = {
    "person": 45, "car": 180, "bus": 250, "truck": 260, "motorcycle": 80,
    "bicycle": 50, "laptop": 35, "cell phone": 7, "bottle": 6, "cup": 8,
    "stop sign": 76
}

class ObjectDetectionApp:
    """
    The main application class for the YOLOv8 Object Detection GUI.
    """
    # --- Color Palette ---
    BG_COLOR = "#212121"              # Dark grey background
    FRAME_COLOR = "#2c3e50"           # Dark blue-grey for frames
    BUTTON_COLOR = "#3498db"          # Bright blue for buttons
    BUTTON_HOVER_COLOR = "#5dade2"    # Lighter blue for hover
    TEXT_COLOR = "#ecf0f1"            # Light grey/white for text
    TITLE_COLOR = "#ffffff"           # White for the main title
    STATUS_SUCCESS_COLOR = "#2ecc71"  # Green for success status
    STATUS_ERROR_COLOR = "#e74c3c"    # Red for error status

    def __init__(self, root_window):
        """Initializes the main application GUI and the YOLOv8 detector."""
        self.root = root_window
        self.root.title("YOLOv8 Object Detection GUI")
        self.root.geometry("500x400")
        self.root.resizable(False, False)
        self.root.configure(bg=self.BG_COLOR)
        
        # Check for model file before starting
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror("Fatal Error", f"Model file not found at '{MODEL_PATH}'.")
            self.root.destroy()
            return
            
        # Initialize the detector
        try:
            self.yolov8_detector = YOLOv8(MODEL_PATH, conf_thres=CONF_THRESHOLD, iou_thres=IOU_THRESHOLD)
        except Exception as e:
            messagebox.showerror("Fatal Error", f"Failed to initialize the YOLOv8 detector: {e}")
            self.root.destroy()
            return

        self.create_widgets()
        self.update_status("Ready", self.STATUS_SUCCESS_COLOR)

    def create_widgets(self):
        """Creates and places all the GUI widgets in the main window."""
        main_frame = tk.Frame(self.root, bg=self.BG_COLOR, padx=20, pady=20)
        main_frame.pack(expand=True, fill=tk.BOTH)

        title_label = tk.Label(main_frame, text="Object Detection Interface", font=("Segoe UI", 22, "bold"), bg=self.BG_COLOR, fg=self.TITLE_COLOR)
        title_label.pack(pady=(0, 30))

        # --- Create styled buttons ---
        self.create_styled_button(main_frame, "🖼️  Detect from Image", self.run_from_image)
        self.create_styled_button(main_frame, "📷  Detect from Webcam", self.run_from_webcam)
        self.create_styled_button(main_frame, "🎞️  Detect from Video File", self.run_from_video)
        self.create_styled_button(main_frame, "📺  Detect from YouTube", self.run_from_youtube)

        # --- Status Bar ---
        self.status_label = tk.Label(self.root, text="", font=("Segoe UI", 10), bg=self.STATUS_SUCCESS_COLOR, fg="#ffffff", anchor='w', padx=10)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def create_styled_button(self, parent, text, command):
        """Helper function to create a styled button with hover effects."""
        button = tk.Button(
            parent,
            text=text,
            command=command,
            font=("Segoe UI", 12, "bold"),
            bg=self.BUTTON_COLOR,
            fg=self.TEXT_COLOR,
            relief=tk.FLAT,
            borderwidth=0,
            activebackground=self.BUTTON_HOVER_COLOR,
            activeforeground=self.TEXT_COLOR,
            pady=10
        )
        button.pack(pady=6, fill=tk.X)

        # Bind hover events
        button.bind("<Enter>", lambda e: e.widget.config(bg=self.BUTTON_HOVER_COLOR))
        button.bind("<Leave>", lambda e: e.widget.config(bg=self.BUTTON_COLOR))
        return button

    def update_status(self, message, color):
        """Updates the status bar with a message and color."""
        self.status_label.config(text=message, bg=color)
        self.root.update_idletasks()

    def run_from_image(self):
        """Handles object detection from a user-selected image file."""
        self.update_status("Opening file dialog...", self.BUTTON_COLOR)
        file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if not file_path:
            self.update_status("Ready", self.STATUS_SUCCESS_COLOR)
            return

        try:
            self.update_status(f"Processing {os.path.basename(file_path)}...", self.BUTTON_COLOR)
            img = cv2.imread(file_path)
            if img is None:
                messagebox.showerror("Error", f"Failed to load the image from {file_path}.")
                self.update_status("Error loading image", self.STATUS_ERROR_COLOR)
                return

            boxes, scores, class_ids = self.yolov8_detector(img)
            combined_img = self.yolov8_detector.draw_detections(img, boxes, scores, class_ids)
            
            self.update_status("Detection complete. Displaying results...", self.STATUS_SUCCESS_COLOR)
            cv2.imshow("Detected Objects - Press any key to close", combined_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            self.update_status("Ready", self.STATUS_SUCCESS_COLOR)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during image processing: {e}")
            self.update_status("Processing error", self.STATUS_ERROR_COLOR)

    def run_from_webcam(self):
        """Handles real-time object detection from a webcam."""
        self.update_status("Starting webcam...", self.BUTTON_COLOR)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not access the webcam.")
            self.update_status("Webcam error", self.STATUS_ERROR_COLOR)
            return
            
        self.update_status("Webcam active. Press 'q' to quit.", self.STATUS_SUCCESS_COLOR)
        cv2.namedWindow("Webcam Detection - Press 'q' to quit", cv2.WINDOW_NORMAL)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            boxes, scores, class_ids = self.yolov8_detector(frame)
            combined_img = self.yolov8_detector.draw_detections(frame, boxes, scores, class_ids, FOCAL_LENGTH, KNOWN_WIDTHS_CM)
            cv2.imshow("Webcam Detection - Press 'q' to quit", combined_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.update_status("Ready", self.STATUS_SUCCESS_COLOR)

    def run_from_video_source(self, cap, source_name="Video"):
        """Generic handler for processing video streams."""
        if not cap or not cap.isOpened():
            messagebox.showerror("Error", f"Failed to open video source: {source_name}.")
            self.update_status(f"Error opening {source_name}", self.STATUS_ERROR_COLOR)
            return
        
        self.update_status(f"Playing {source_name}. Press 'q' to quit.", self.STATUS_SUCCESS_COLOR)
        window_title = f"{source_name} Detection - Press 'q' to quit"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            start_time = time.time()
            boxes, scores, class_ids = self.yolov8_detector(frame)
            combined_img = self.yolov8_detector.draw_detections(frame, boxes, scores, class_ids)
            
            end_time = time.time()
            fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0
            cv2.putText(combined_img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(window_title, combined_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.update_status("Ready", self.STATUS_SUCCESS_COLOR)

    def run_from_video(self):
        """Handles detection from a local video file."""
        self.update_status("Opening file dialog...", self.BUTTON_COLOR)
        file_path = filedialog.askopenfilename(title="Select a Video File", filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv")])
        if not file_path:
            self.update_status("Ready", self.STATUS_SUCCESS_COLOR)
            return
        cap = cv2.VideoCapture(file_path)
        self.run_from_video_source(cap, source_name=os.path.basename(file_path))

    def run_from_youtube(self):
        """Handles detection from a YouTube video URL."""
        if cap_from_youtube is None:
            messagebox.showerror("Error", "Module not found: 'cap_from_youtube'.\nPlease run: pip install cap-from-youtube")
            self.update_status("Dependency error", self.STATUS_ERROR_COLOR)
            return
            
        self.update_status("Waiting for YouTube URL...", self.BUTTON_COLOR)
        url = simpledialog.askstring("YouTube URL", "Enter YouTube Video URL:", parent=self.root)
        
        if url:
            try:
                cap = cap_from_youtube(url, "720p")
                self.run_from_video_source(cap, source_name="YouTube")
            except Exception as e:
                messagebox.showerror("Error", f"Could not open YouTube stream.\nError: {e}")
                self.update_status("YouTube stream error", self.STATUS_ERROR_COLOR)
        else:
            self.update_status("Ready", self.STATUS_SUCCESS_COLOR)

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
