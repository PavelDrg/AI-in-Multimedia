# Import necessary libraries
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import os
import pyttsx3
from moviepy.editor import VideoFileClip, AudioFileClip

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
IMAGE_FILE = 'image.png'
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

# WINDOW_WIDTH = 800
# WINDOW_HEIGHT = 800
#
# # Display the input image, GrabCut mask, and segmented object
# resized_img = cv2.resize(img, (WINDOW_WIDTH, WINDOW_HEIGHT))
# resized_rgb_annotated_image = cv2.resize(rgb_annotated_image, (WINDOW_WIDTH, WINDOW_HEIGHT))
# resized_grabcut_mask = cv2.resize(grabcut_mask, (WINDOW_WIDTH, WINDOW_HEIGHT))
# resized_segmented_object = cv2.resize(segmented_object, (WINDOW_WIDTH, WINDOW_HEIGHT))
# cv2.imshow("Input Image", resized_img)
# cv2.imshow('Detected Object Image', resized_rgb_annotated_image)
# cv2.imshow("Binary Mask", resized_grabcut_mask)
# cv2.imshow("Extracted Object", resized_segmented_object)
# cv2.waitKey(0)

print("Imagini prelucrate")

# Constants
fps = 25.0
duration = 5  # seconds
dimensions = (1900, 1900)

# Define the list of image paths
image_paths = [
    'image.png',
    'object_detected_img.png',
    'extracted_object.png',
    'binary_mask.png',
    'yuv_img.tiff',
    'yuv_extracted_object.tiff'
]

# Define custom durations for each image (in seconds)
custom_durations = [3, 2.5, 2, 2, 5, 5]  # Modify these values as needed

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

output_video = cv2.VideoWriter('video.avi', fourcc, fps, dimensions)

# Define the text parameters
font = cv2.FONT_HERSHEY_SIMPLEX
text_position = (100, 100)  # Position of the text in the frame
font_scale = 2
font_color = (255, 255, 255)  # White color
thickness = 2

# Define the text to be displayed for each image
image_texts = ['Original image with black surfboard on the beach', 'Image with surfboard detected', 'Extracted object', 'Mask with the object', 'Original image in YUV color space saved as tiff file', 'Mask with object in YUV colorspace saved as tiff file']

# OpenCV VideoWriter doesn't support transparent text overlay, so we'll create a black background
black_frame = np.zeros((dimensions[1], dimensions[0], 3), dtype=np.uint8)

# Write each image to the video with text overlay
for image_path, text, duration in zip(image_paths, image_texts, custom_durations):
    # Read the image
    img = cv2.imread(image_path)

    # Check if the image was successfully read
    if img is None:
        print(f"Error: Unable to read image from path: {image_path}")
        continue

    # Resize the image to fit within the frame
    img_resized = cv2.resize(img, (1560, 1560))  # Adjust dimensions as needed

    # Add text to the frame
    frame_with_text = black_frame.copy()
    # Split the text into lines
    lines = text.split('\n')

    # Draw each line of text
    y_offset = text_position[1]
    for line in lines:
        cv2.putText(frame_with_text, line, (text_position[0], y_offset), font, font_scale, font_color, thickness,
                    cv2.LINE_AA)
        y_offset += int(18 * font_scale * thickness)

    # Calculate the position to place the resized image
    x_offset = (frame_with_text.shape[1] - img_resized.shape[1]) // 2
    y_offset = (frame_with_text.shape[0] - img_resized.shape[0]) // 2

    # Combine the image with the frame containing the text
    frame_with_text[y_offset:y_offset + img_resized.shape[0], x_offset:x_offset + img_resized.shape[1]] = img_resized

    # Write the frame to the video with custom duration
    for _ in range(int(fps * duration)):
        output_video.write(frame_with_text)

# Release the video writer
output_video.release()
print("Video Saved")

# Initialize the pyttsx3 engine
engine = pyttsx3.init()

# Set properties (optional)
engine.setProperty('rate', 150)  # Speed of speech


# Define the texts to be spoken
text = ('Original image with black surfboard on the beach. Image with surfboard detected. Extracted object. Mask '
        'with the object. Original image in YUV color space saved as tiff file. Mask with object in YUV colorspace '
        'saved as tiff file.')

# Define the output audio file name
output_file = "audio.mp3"

# Iterate through the texts
engine.save_to_file(text, output_file)  # Save text to the output file

engine.runAndWait()

print("Audio inregistrat")

# Load the video and audio files
video = VideoFileClip("video.avi")
audio = AudioFileClip("audio.mp3")

# Set the audio of the video to the audio from audio.mp3
video = video.set_audio(audio)

# Write the modified video with the new audio using the DivX codec
video.write_videofile("video_with__audio.avi", codec="libxvid", audio_codec="mp3")

# Optional: If you want to overwrite the old video with the new one
#os.replace("video_with_audio.avi", "video.avi")

print("Clip final salvat")
