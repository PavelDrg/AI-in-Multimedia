import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import os
import pyttsx3
from moviepy.editor import VideoFileClip, AudioFileClip

from tensorflow import keras
import  keras_cv
import matplotlib.pyplot as plt
from PIL import Image

# --------------------------------------------------------
# GENERAREA DE IMAGINI

keras.mixed_precision.set_global_policy("mixed_float16")

# Crearea modelului
model = keras_cv.models.StableDiffusion(img_height=512,
                                        img_width=512,
                                        jit_compile=True)

def plot_images(images):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")

# Aici dam prompt-ul in functie de asignare
images = model.text_to_image(prompt="Realistic beach scene with a black surfboard as the main subject and the light blue sea in the background.",
                             batch_size=1)

plot_images(images)

# --------------------------------------------------------
# PRELUCRAREA IMAGINII ALESE

# Constante pentru partea de detectie de obiect (modificam in caz ca vrem sa arate altfel incercuirea obiectului)
MARGIN = 10  # pixeli
ROW_SIZE = 10  # pixeli
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (0, 255, 0)


# Vizualizam bounding box pe imaginea originala
def visualize(image, detection_result):
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)
    return image


# Functie de creare a mastii binare cerute folosind GrabCut
def grabcut_segmentation(image, rect, iter_count=10):
    # Initializam masca
    mask = np.zeros(image.shape[:2], dtype="uint8")

    fg_model = np.zeros((1, 65), dtype="float")
    bg_model = np.zeros((1, 65), dtype="float")
    # Aplicam GrabCut
    (mask, bg_model, fg_model) = cv2.grabCut(image, mask, rect, bg_model, fg_model,
                                             iterCount=iter_count, mode=cv2.GC_INIT_WITH_RECT)
    # Generam masca binara
    output_mask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1)
    # Scalam masca
    output_mask = (output_mask * 255).astype("uint8")
    # Aplicam un AND bit cu bit ca sa generam obiectul extras
    segmented_object = cv2.bitwise_and(image, image, mask=output_mask)
    return output_mask, segmented_object


# Citim imaginea originala
IMAGE_FILE = 'image.png'
img = cv2.imread(IMAGE_FILE)

# Cream modelul de detectare de obiecte
base_options = python.BaseOptions(model_asset_path='efficientdet_lite0.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

# Incarcam imaginea originala
image = mp.Image.create_from_file(IMAGE_FILE)

# Detectam obiectul
detection_result = detector.detect(image)

# Procesam rezultatul detectiei
image_copy = np.copy(image.numpy_view())
annotated_image = visualize(image_copy, detection_result)
rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

# Luam coordonatele bounding box-ului
for detection in detection_result.detections:
    bbox = detection.bounding_box
    rect = (int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height))
    break

# Aplicam GrabCut folosind coordonatele de la detectie
grabcut_mask, segmented_object = grabcut_segmentation(rgb_annotated_image, rect)

output_folder = os.path.dirname(__file__)

cv2.imwrite(os.path.join(output_folder, 'object_detected_img.png'), rgb_annotated_image)
cv2.imwrite(os.path.join(output_folder, 'binary_mask.png'), grabcut_mask)
cv2.imwrite(os.path.join(output_folder, 'extracted_object.png'), segmented_object)

# Convertim imaginea originala in spatiul de culoare YUV
yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
cv2.imwrite(os.path.join(output_folder, 'yuv_img.tiff'), yuv_img)

# Convertim obiectul extras in spatiul de culoare YUV
yuv_segmented_object = cv2.cvtColor(segmented_object, cv2.COLOR_BGR2YUV)
cv2.imwrite(os.path.join(output_folder, 'yuv_extracted_object.tiff'), yuv_segmented_object)

# In caz ca vrem sa fie afisate imaginile prelucrate nu doar salvate decomentam codul de mai jos

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

# --------------------------------------------------------
# CREAREA PRIMULUI VIDEOCLIP

# Constante pentru videoclip
fps = 25.0
duration = 5  # seconds
dimensions = (1900, 1900)

# Caile catre imaginile folosite
image_paths = [
    'image.png',
    'object_detected_img.png',
    'extracted_object.png',
    'binary_mask.png',
    'yuv_img.tiff',
    'yuv_extracted_object.tiff'
]

# Listam pentru cate secunde vrem sa apara fiecare imagine
custom_durations = [3, 2.5, 2, 2, 5, 5]

# Initializam video writer (introducem aici si codecul dorit)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# fourcc = cv2.VideoWriter_fourcc(*'3IVD')
# fourcc = cv2.VideoWriter_fourcc(*'DX50')

output_video = cv2.VideoWriter('video.avi', fourcc, fps, dimensions)

# Proprietatile textului din clip
font = cv2.FONT_HERSHEY_SIMPLEX
text_position = (100, 100)
font_scale = 2
font_color = (255, 255, 255)
thickness = 2

# Textul corespunzator fiecarei imagini
image_texts = ['Original image with black surfboard on the beach', 'Image with surfboard detected', 'Extracted object', 'Mask with the object', 'Original image in YUV color space saved as tiff file', 'Mask with object in YUV colorspace saved as tiff file']

# Facem un fundal pentru text
black_frame = np.zeros((dimensions[1], dimensions[0], 3), dtype=np.uint8)

# Punem fiecare imagine in video impreuna cu textul
for image_path, text, duration in zip(image_paths, image_texts, custom_durations):
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Unable to read image from path: {image_path}")
        continue

    img_resized = cv2.resize(img, (1560, 1560))

    frame_with_text = black_frame.copy()
    cv2.putText(frame_with_text, text, text_position, font, font_scale, font_color, thickness, cv2.LINE_AA)

    x_offset = (frame_with_text.shape[1] - img_resized.shape[1]) // 2
    y_offset = (frame_with_text.shape[0] - img_resized.shape[0]) // 2

    frame_with_text[y_offset:y_offset + img_resized.shape[0], x_offset:x_offset + img_resized.shape[1]] = img_resized

    for _ in range(int(fps * duration)):
        output_video.write(frame_with_text)

# Finalizam clipul
output_video.release()
print("Video Saved")

# --------------------------------------------------------
# CREAREA INREGISTRARII AUDIO

# Pornim engine-ul pentru inregistrarea audio
engine = pyttsx3.init()

# Proprietatile vocii folosite (am modificat doar rate of speech-ul)
engine.setProperty('rate', 150)


# Introducem aici textul dorit
text = ('Original image with black surfboard on the beach. Image with surfboard detected. Extracted object. Mask '
        'with the object. Original image in YUV color space saved as tiff file. Mask with object in YUV colorspace '
        'saved as tiff file.')

# Cum e salvata inregistrarea
output_file = "audio.mp3"

# Salvam in fisier
engine.save_to_file(text, output_file)

engine.runAndWait()

print("Audio inregistrat")

# --------------------------------------------------------
# CREAREA VIDEOCLIPULUI CU SONOR

# Incarcam fisierul audio si video
video = VideoFileClip("video.avi")
audio = AudioFileClip("audio.mp3")

# Punem audioul in video
video = video.set_audio(audio)

# Scriem fisierul video modificat (putem schimba codecul)
video.write_videofile("video_with__audio.avi", codec="libxvid", audio_codec="mp3")

# In caz ca vrem sa avem doar videoclipul cu sunet decomentam urmatoarea linie
#os.replace("video_with_audio.avi", "video.avi")

print("Clip final salvat")
