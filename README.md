# **YOLOv8 Object Detection GUI**

---

### **1. Introduction**
This project provides a comprehensive and user-friendly interface for real-time object detection using the YOLOv8 model, powered by ONNX Runtime. It supports detection from various sources including images, webcam feeds, local video files, and even YouTube video streams. The application features a Tkinter-based GUI for intuitive control and includes basic distance estimation for objects detected via webcam.

---

### **2. Features**
* **YOLOv8 ONNX Inference:** Leverages a pre-trained YOLOv8m model optimized for ONNX Runtime for efficient and fast detections.
* **Tkinter GUI:** A clean and intuitive graphical user interface (GUI) for easy interaction.
* **Multiple Input Sources:**
    * **Image Detection:** Analyze objects in static image files.
    * **Webcam Detection:** Real-time object detection from your local webcam feed, including optional distance estimation.
    * **Local Video File Detection:** Process objects in video files from your computer.
    * **YouTube Stream Detection:** Detect objects directly from specified YouTube video URLs (requires `cap-from-youtube` library).
* **Bounding Box & Labeling:** Draws bounding boxes, confidence scores, and class labels on detected objects.
* **FPS Counter:** Displays frames per second during video processing.
* **Modular Design:** Separates the YOLOv8 inference logic into a dedicated module for maintainability.

---

### **3. Technologies Used**
This project utilizes the following key technologies:

* **Programming Language:**
    * Python 3.x
* **Core Libraries/Frameworks:**
    * **OpenCV (`cv2`):** For image and video processing, webcam access, and drawing detections.
    * **NumPy:** For numerical operations and data handling.
    * **ONNX Runtime:** For efficient inference of the YOLOv8 ONNX model, supporting both CPU and GPU.
    * **Tkinter:** Python's standard GUI library, used for the main application interface.
    * **`cap-from-youtube`:** (Optional) For streaming YouTube videos.
* **Model:**
    * **YOLOv8m:** A state-of-the-art object detection model.

---

### **4. Getting Started**
Follow these instructions to set up and run the project on your local machine.

#### **4.1. Prerequisites**
Ensure you have the following software installed:

* **Python 3.x:** Download from [python.org](https://www.python.org/). It's recommended to use Python 3.8 or newer.
* **Git:** Download from [git-scm.com](https://git-scm.com/downloads).
* **FFmpeg (Optional but Recommended):** For robust video processing and fixing potential codec issues with downloaded videos or streams. Download from [ffmpeg.org](https://ffmpeg.org/download.html) and ensure it's added to your system's PATH.

#### **4.2. Installation**

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[your-username]/[your-repo-name].git
    cd [your-repo-name]
    ```
    (Replace `[your-username]` and `[your-repo-name]` with your actual GitHub details)

2.  **Create a virtual environment (highly recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    * **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    * **For GPU acceleration (if you have an NVIDIA GPU):**
        Instead of `onnxruntime` in `requirements.txt`, install `onnxruntime-gpu` manually *after* installing other dependencies:
        ```bash
        pip install onnxruntime-gpu
        ```
        (Make sure your CUDA Toolkit and cuDNN are correctly installed and configured for PyTorch/ONNX Runtime).

#### **4.3. Download the YOLOv8 Model**
The pre-trained YOLOv8m ONNX model is not included in this repository due to its file size. You need to download it manually:

1.  Go to the official Ultralytics YOLOv8 releases page or a reliable source.
2.  Download the `yolov8m.onnx` model file.
    * A common download link might look like this (check for the latest version): [https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8m.onnx](https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8m.onnx)
3.  Place the downloaded `yolov8m.onnx` file directly into the **root directory** of this project (i.e., `[your-repo-name]/yolov8m.onnx`).

---

### **5. Usage**

The primary way to interact with this project is through its graphical user interface.

#### **5.1. Running the GUI Application (Recommended)**

1.  **Activate your virtual environment** (if not already active):
    * **Windows:** `.\venv\Scripts\activate`
    * **macOS/Linux:** `source venv/bin/activate`

2.  **Navigate to the root of the project directory:**
    ```bash
    cd [your-repo-name]
    ```

3.  **Run the GUI application:**
    ```bash
    python app_gui.py
    ```
    A GUI window will appear, providing buttons to:
    * **Detect from Image:** Select an image file to perform object detection.
    * **Detect from Webcam:** Start real-time object detection using your webcam.
    * **Detect from Video File:** Select a local video file for object detection.
    * **Detect from YouTube:** Enter a YouTube video URL to stream and detect objects.

#### **5.2. Running Standalone Scripts (Advanced/Debugging)**

You can also run individual detection scripts directly from the terminal for specific tasks:

1.  **Activate your virtual environment.**
2.  **Navigate to the root of the project directory.**

3.  **Image Detection:**
    * Make sure `input_image.jpg` is in the root directory or update the `image_path` in the script.
    ```bash
    python image_object_detection.py
    ```

4.  **Webcam Detection:**
    ```bash
    python object_detection_webcam.py
    ```

5.  **Local Video File Detection:**
    ```bash
    python object_detection_downloaded_video.py
    ```
    (A file dialog will open to select your video.)

6.  **YouTube Video Stream Detection:**
    ```bash
    python object_detection_youtube.py
    ```
    (Requires `cap-from-youtube` and a valid YouTube URL configured in the script or provided via prompt.)

---

### **6. Screenshots/Demos**
*(Replace these with actual links to your images or GIFs)*

* **GUI Main Window:** A clear shot of the main application interface.
    ![GUI Main Window](https://via.placeholder.com/700x400?text=Object+Detection+GUI)

* **Webcam Detection in Action:** Showcase real-time detection on a webcam feed, possibly demonstrating distance estimation.
    ![Webcam Detection Demo](https://via.placeholder.com/700x400?text=Webcam+Object+Detection)

* **Image Detection Example:** An example of objects detected in a static image.
    ![Image Detection Example](https://via.placeholder.com/700x400?text=Image+Object+Detection+Result)

* **Video Detection Example:** A screenshot or short GIF from a video detection session.
    ![Video Detection Example](https://via.placeholder.com/700x400?text=Video+Object+Detection)

---

### **7. Project Structure**
object_detection/
├── .gitignore             # Files and directories to ignore for Git
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── yolov8/                # Contains the core YOLOv8 inference class
│   └── YOLOv8.py          # The class for loading and running YOLOv8 ONNX model
├── app_gui.py             # The main Tkinter GUI application script
├── image_object_detection.py # Standalone script for image detection
├── input_image.jpg        # Sample input image
├── instructions.txt       # Any specific instructions or notes
├── object_detection_downloaded_video.py # Standalone script for local video detection
├── object_detection_webcam.py # Standalone script for webcam detection
├── object_detection_youtube.py # Standalone script for YouTube stream detection
└── yolov8m.onnx           # (Download Separately) The YOLOv8m ONNX model file
