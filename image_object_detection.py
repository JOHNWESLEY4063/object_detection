import cv2
import numpy as np
import onnxruntime as ort
import os

# Configurations
model_path = "C:\\Users\\peram\\object_detection\\yolov8-onnx-env\\Scripts\\yolov8m.onnx"  # Update this if the model is in a different location
image_path = "C:\\Users\\peram\\object_detection\\input_image.jpg"     # Update this to the correct image path if different

# Ensure the model file exists
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found! Please place 'yolov8m.onnx' in the current directory or update model_path.")
    exit(1)

# Load ONNX model with GPU support
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]
session = ort.InferenceSession(model_path, providers=providers)
print(f"✅ ONNX model loaded successfully. Running on: {providers[0]}")

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

# Get model input size
model_input_shape = session.get_inputs()[0].shape  # Typically (1, 3, 640, 640)
model_width, model_height = model_input_shape[2], model_input_shape[3]

# Read the local image
img = cv2.imread(image_path)

# Check if image is loaded successfully
if img is None:
    print(f"Error: Failed to load image '{image_path}'. Check file path or integrity.")
    exit(1)
print(f"✅ Image '{image_path}' loaded successfully.")

# Preprocess the image for YOLO model
orig_h, orig_w, _ = img.shape
image_resized = cv2.resize(img, (model_width, model_height))
image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
image_input = image_rgb.astype(np.float32) / 255.0  # Normalize
image_input = np.transpose(image_input, (2, 0, 1))  # HWC to CHW
image_input = np.expand_dims(image_input, axis=0)  # Add batch dimension

# Run inference
inputs = {session.get_inputs()[0].name: image_input}
outputs = session.run(None, inputs)[0]  # YOLOv8 ONNX output shape: (1, 84, 8400)

# Post-process detections
detections = outputs[0]  # Shape: (84, 8400)
boxes = []
scores = []
class_ids = []

for i in range(detections.shape[1]):
    detection = detections[:, i]
    x_center, y_center, width, height = detection[0:4]
    class_scores = detection[4:]
    class_id = np.argmax(class_scores)
    score = class_scores[class_id]

    if score > 0.2:  # Confidence threshold from original code
        x1 = int((x_center - width / 2) * orig_w / model_width)
        y1 = int((y_center - height / 2) * orig_h / model_height)
        x2 = int((x_center + width / 2) * orig_w / model_width)
        y2 = int((y_center + height / 2) * orig_h / model_height)

        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(orig_w, x2), min(orig_h, y2)
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(float(score))
        class_ids.append(class_id)

# Apply Non-Max Suppression (using a simple threshold-based approach)
def non_max_suppression(boxes, scores, iou_threshold=0.3):
    if len(boxes) == 0:
        return []
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.2, iou_threshold).flatten()
    return indices

indices = non_max_suppression(boxes, scores, 0.3)

# Draw detections on image
for idx in indices:
    x, y, w, h = boxes[idx]
    x1, y1, x2, y2 = x, y, x + w, y + h
    score = scores[idx]
    class_id = class_ids[idx]

    class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"Unknown {class_id}"
    label = f"{class_name}: {score:.2f}"

    # Draw green bounding box
    color = (0, 255, 0)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # Display label with a background
    font_scale = 0.5
    thickness = 1
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, (x1, y1 - text_h - 5), (x1 + text_w + 5, y1), color, -1)
    cv2.putText(img, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

# Ensure output directory exists
output_dir = "doc/img/"
os.makedirs(output_dir, exist_ok=True)

# Save the output image
output_path = os.path.join(output_dir, "detected_objects.jpg")
cv2.imwrite(output_path, img)
print(f"✅ Detection results saved to {output_path}")

# Show the result
cv2.imshow("Detected Objects", img)
cv2.waitKey(0)
cv2.destroyAllWindows()