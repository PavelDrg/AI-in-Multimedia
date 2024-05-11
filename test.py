import pyttsx3
import time

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