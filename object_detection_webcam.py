import cv2
import numpy as np
import onnxruntime as ort
import time

# Load COCO class names
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Load ONNX model with GPU support
onnx_model_path = "C:\\Users\\peram\\object_detection\\yolov8-onnx-env\\Scripts\\yolov8m.onnx"
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]
session = ort.InferenceSession(onnx_model_path, providers=providers)

print(f"ONNX model loaded successfully. Running on: {providers[0]}")

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Error: Could not access the webcam.")

# Get model input size
model_input_shape = session.get_inputs()[0].shape  # Typically (1, 3, 640, 640)
model_width, model_height = model_input_shape[2], model_input_shape[3]

# FPS Counter
fps_start_time = time.time()
frame_count = 0
conf_threshold = 0.25  # Adjusted for better detection
nms_threshold = 0.45  # Non-Max Suppression threshold

def non_max_suppression(boxes, scores, iou_threshold):
    """Apply Non-Max Suppression to filter overlapping boxes."""
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)
    if len(indices) > 0:
        indices = indices.flatten()
    return indices

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    orig_h, orig_w, _ = frame.shape  # Get frame size

    # Preprocess the frame for YOLO model
    image_resized = cv2.resize(frame, (model_width, model_height))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_input = image_rgb.astype(np.float32) / 255.0  # Normalize
    image_input = np.transpose(image_input, (2, 0, 1))  # HWC to CHW
    image_input = np.expand_dims(image_input, axis=0)  # Add batch dimension

    # Run inference
    inputs = {session.get_inputs()[0].name: image_input}
    outputs = session.run(None, inputs)[0]  # YOLOv8 ONNX output shape: (1, 84, 8400)

    # Post-process detections
    detections = outputs[0]  # Shape: (84, 8400) -> 8400 detections, each with 84 values (4 bbox + 80 class scores)

    boxes = []
    scores = []
    class_ids = []
    detected_objects = []

    for i in range(detections.shape[1]):  # Loop over each detection
        detection = detections[:, i]
        # First 4 values are bounding box coordinates (center_x, center_y, width, height)
        x_center, y_center, width, height = detection[0:4]
        # Remaining values are class scores
        class_scores = detection[4:]
        class_id = np.argmax(class_scores)
        score = class_scores[class_id]

        if score > conf_threshold:  # Confidence filter
            # Convert center-based coordinates to top-left and bottom-right
            x1 = int((x_center - width / 2) * orig_w / model_width)
            y1 = int((y_center - height / 2) * orig_h / model_height)
            x2 = int((x_center + width / 2) * orig_w / model_width)
            y2 = int((y_center + height / 2) * orig_h / model_height)

            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(orig_w, x2), min(orig_h, y2)  # Ensure within bounds

            # Store for NMS
            boxes.append([x1, y1, x2 - x1, y2 - y1])  # Format: [x, y, w, h]
            scores.append(float(score))
            class_ids.append(class_id)

    # Apply Non-Max Suppression to remove overlapping boxes
    indices = non_max_suppression(boxes, scores, nms_threshold)

    # Draw bounding boxes and labels for remaining detections
    for idx in indices:
        x, y, w, h = boxes[idx]
        x1, y1, x2, y2 = x, y, x + w, y + h
        score = scores[idx]
        class_id = class_ids[idx]

        # Get class label
        class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"Unknown {class_id}"
        label = f"{class_name}: {score:.2f}"

        # Store detected objects for debugging
        detected_objects.append((label, score))

        # Draw green bounding box
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Display label with a background
        font_scale = 0.7
        thickness = 2
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_w, text_h = text_size

        # Draw background rectangle for label
        cv2.rectangle(frame, (x1, y1 - text_h - 5), (x1 + text_w + 10, y1), color, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    # Debugging: Print detected objects in console
    print("Detected Objects:", detected_objects)

    # FPS Calculation
    frame_count += 1
    fps_end_time = time.time()
    fps = frame_count / (fps_end_time - fps_start_time)

    # Display FPS
    fps_text = f"FPS: {fps:.2f} | Conf: {conf_threshold:.2f}"
    cv2.rectangle(frame, (10, 5), (250, 35), (0, 255, 0), -1)
    cv2.putText(frame, fps_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Show result
    cv2.imshow("Webcam Object Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()