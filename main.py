import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from skimage import io, img_as_float
from skimage.transform import resize
import os
import glob

# Function to load and preprocess a single image
def preprocess_image(img_path, image_size=(400, 600)):
    img = img_as_float(io.imread(img_path))
    img = resize(img, image_size)
    return img

# Function to save a single image
def save_image(image, path):
    io.imsave(path, (image * 255).astype(np.uint8))

# Load the fine-tuned denoising model
model_path = 'denoising_fine_tuned_model.h5'
model = load_model(model_path, custom_objects={'psnr': tf.image.psnr})

# Define paths
test_low_folder = '/test/low'
test_pred_folder = '/test/predicted'

# Ensure the output directory exists
os.makedirs(test_pred_folder, exist_ok=True)

# Process each test image
test_image_paths = sorted(glob.glob(os.path.join(test_low_folder, '*.png')))
for img_path in test_image_paths:
    # Load and preprocess the image
    img = preprocess_image(img_path)

    # Predict the denoised image
    img_input = np.expand_dims(img, axis=0)
    denoised_img = model.predict(img_input)[0]

    # Save the denoised image
    base_name = os.path.basename(img_path)
    save_image(denoised_img, os.path.join(test_pred_folder, base_name))

print("Denoised images saved to:", test_pred_folder)
