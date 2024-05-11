# Import necessary libraries
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import os

# Constants
MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (0, 255, 0)  # green


# Function to visualize bounding boxes on the input image
def visualize(image, detection_result):
    """Draws bounding boxes on the input image and return it."""
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)
    return image


# Function to create a binary mask using GrabCut
def grabcut_segmentation(image, rect, iter_count=10):
    # Initialize mask
    mask = np.zeros(image.shape[:2], dtype="uint8")
    # Initialize model arrays
    fg_model = np.zeros((1, 65), dtype="float")
    bg_model = np.zeros((1, 65), dtype="float")
    # Apply GrabCut
    (mask, bg_model, fg_model) = cv2.grabCut(image, mask, rect, bg_model, fg_model,
                                             iterCount=iter_count, mode=cv2.GC_INIT_WITH_RECT)
    # Generate binary mask
    output_mask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1)
    # Scale the mask
    output_mask = (output_mask * 255).astype("uint8")
    # Apply bitwise AND to generate segmented object
    segmented_object = cv2.bitwise_and(image, image, mask=output_mask)
    return output_mask, segmented_object


# Load input image
IMAGE_FILE = 'img.png'
img = cv2.imread(IMAGE_FILE)

# Create ObjectDetector object
base_options = python.BaseOptions(model_asset_path='efficientdet_lite0.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

# Load the input image
image = mp.Image.create_from_file(IMAGE_FILE)

# Detect objects in the input image
detection_result = detector.detect(image)

# Process the detection result
image_copy = np.copy(image.numpy_view())
annotated_image = visualize(image_copy, detection_result)
rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

# Extract bounding box coordinates from the first program
for detection in detection_result.detections:
    bbox = detection.bounding_box
    rect = (int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height))
    break  # Assuming there's only one object

# Apply GrabCut using the bounding box segmentation method
grabcut_mask, segmented_object = grabcut_segmentation(rgb_annotated_image, rect)

output_folder = os.path.dirname(__file__)

cv2.imwrite(os.path.join(output_folder, 'object_detected_img.png'), rgb_annotated_image)
cv2.imwrite(os.path.join(output_folder, 'binary_mask.png'), grabcut_mask)
cv2.imwrite(os.path.join(output_folder, 'extracted_object.png'), segmented_object)

# Convert input image to YUV color space
yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
cv2.imwrite(os.path.join(output_folder, 'yuv_img.tiff'), yuv_img)

# Convert segmented object to YUV color space
yuv_segmented_object = cv2.cvtColor(segmented_object, cv2.COLOR_BGR2YUV)
cv2.imwrite(os.path.join(output_folder, 'yuv_extracted_object.tiff'), yuv_segmented_object)

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800

# Display the input image, GrabCut mask, and segmented object
resized_img = cv2.resize(img, (WINDOW_WIDTH, WINDOW_HEIGHT))
resized_rgb_annotated_image = cv2.resize(rgb_annotated_image, (WINDOW_WIDTH, WINDOW_HEIGHT))
resized_grabcut_mask = cv2.resize(grabcut_mask, (WINDOW_WIDTH, WINDOW_HEIGHT))
resized_segmented_object = cv2.resize(segmented_object, (WINDOW_WIDTH, WINDOW_HEIGHT))
# cv2.imshow("Input Image", resized_img)
# cv2.imshow('Detected Object Image', resized_rgb_annotated_image)
# cv2.imshow("Binary Mask", resized_grabcut_mask)
# cv2.imshow("Extracted Object", resized_segmented_object)
# cv2.waitKey(0)


# Constants
COMPRESSED_FORMAT = True
fps = 25.0
duration_per_image = 5  # seconds
dimensions = (800, 800)

# Define the list of image paths
image_paths = [
    'img.png',
    'object_detected_img.png',
    'extracted_object.png',
    'binary_mask.png',
    'yuv_img.tiff',
    'yuv_extracted_object.tiff'
]

# Initialize video writer
if COMPRESSED_FORMAT:
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
else:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

output_video = cv2.VideoWriter('output.avi', fourcc, fps, dimensions)

# Iterate over each image
for image_path in image_paths:
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        continue

    # Resize the image to match video dimensions
    resized_image = cv2.resize(image, dimensions)

    # Write the image to the video for the specified duration
    for _ in range(int(fps * duration_per_image)):
        output_video.write(resized_image)

# Release the video writer
output_video.release()
print("Video Saved")
