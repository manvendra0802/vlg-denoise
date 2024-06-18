import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, ReLU, Add, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from skimage import io, img_as_float
import os
import glob
from skimage.transform import resize

# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Function to apply radial crop
def radial_crop(image, crop_size, image_size=(400, 600)):
    center_x, center_y = image_size[1] // 2, image_size[0] // 2
    radius = min(center_x, center_y, crop_size // 2)
    mask = np.zeros((image_size[0], image_size[1]), dtype=bool)
    Y, X = np.ogrid[:image_size[0], :image_size[1]]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    mask[dist_from_center <= radius] = True
    cropped_image = np.zeros_like(image)
    cropped_image[mask] = image[mask]
    return resize(cropped_image, (crop_size, crop_size))

# Define a simple Residual Block
def residual_block(x, filters, kernel_size=(3, 3)):
    y = Conv2D(filters, kernel_size, padding='same')(x)
    y = BatchNormalization()(y)
    y = ReLU()(y)
    y = Conv2D(filters, kernel_size, padding='same')(y)
    y = BatchNormalization()(y)
    return Add()([x, y])

# Define the denoising model
def build_denoising_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), padding='same')(inputs)
    x = ReLU()(x)

    for _ in range(8):  # Number of residual blocks
        x = residual_block(x, 64)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = ReLU()(x)
    outputs = Conv2D(3, (3, 3), padding='same')(x)
    outputs = Add()([inputs, outputs])

    return Model(inputs, outputs)

# Load and preprocess a single image
def preprocess_image(img_path, crop_size, image_size=(400, 600)):
    img = img_as_float(io.imread(img_path))
    img = resize(img, image_size)
    img = radial_crop(img, crop_size, image_size)
    return img

# Data generator for batch processing with augmentation
def data_generator(low_folder, high_folder, crop_size, image_size=(400, 600), batch_size=32):
    low_image_paths = sorted(glob.glob(os.path.join(low_folder, '*.png')))
    high_image_paths = sorted(glob.glob(os.path.join(high_folder, '*.png')))

    data_gen_args = dict(horizontal_flip=True,
                         vertical_flip=True,
                         rotation_range=90)

    image_datagen = ImageDataGenerator(**data_gen_args)

    while True:
        for i in range(0, len(low_image_paths), batch_size):
            low_batch_paths = low_image_paths[i:i + batch_size]
            high_batch_paths = high_image_paths[i:i + batch_size]

            low_images = np.array([preprocess_image(p, crop_size, image_size) for p in low_batch_paths])
            high_images = np.array([preprocess_image(p, crop_size, image_size) for p in high_batch_paths])

            yield low_images, high_images

# Data generator for full-size images
def data_generator_full(low_folder, high_folder, image_size=(400, 600), batch_size=32):
    low_image_paths = sorted(glob.glob(os.path.join(low_folder, '*.png')))
    high_image_paths = sorted(glob.glob(os.path.join(high_folder, '*.png')))

    data_gen_args = dict(horizontal_flip=True,
                         vertical_flip=True,
                         rotation_range=90)

    image_datagen = ImageDataGenerator(**data_gen_args)

    while True:
        for i in range(0, len(low_image_paths), batch_size):
            low_batch_paths = low_image_paths[i:i + batch_size]
            high_batch_paths = high_image_paths[i:i + batch_size]

            low_images = np.array([resize(img_as_float(io.imread(p)), image_size) for p in low_batch_paths])
            high_images = np.array([resize(img_as_float(io.imread(p)), image_size) for p in high_batch_paths])

            yield low_images, high_images

# Custom PSNR metric
def psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

# Training function with learning rate scheduling
def train_model(low_folder, high_folder, val_low_folder, val_high_folder, crop_size, input_shape, image_size=(400, 600), epochs=100, batch_size=16, fine_tune=False, model_path=None):
    if fine_tune and model_path:
        model = load_model(model_path, custom_objects={'psnr': psnr})
    else:
        model = build_denoising_model(input_shape)

    model.compile(optimizer=Adam(1e-4), loss=MeanAbsoluteError(), metrics=[psnr])

    steps_per_epoch = len(glob.glob(os.path.join(low_folder, '*.png'))) // batch_size
    validation_steps = len(glob.glob(os.path.join(val_low_folder, '*.png'))) // batch_size

    # Learning rate scheduler
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 0.9 ** (epoch // 10))

    # Model checkpointing to save best model
    checkpoint = tf.keras.callbacks.ModelCheckpoint('denoising_best_model.h5', monitor='val_psnr', mode='max', save_best_only=True, save_weights_only=False)

    if fine_tune:
        train_ds = tf.data.Dataset.from_generator(
            lambda: data_generator_full(low_folder, high_folder, image_size, batch_size),
            output_types=(tf.float32, tf.float32),
            output_shapes=((None, image_size[0], image_size[1], 3), (None, image_size[0], image_size[1], 3))
        )

        val_ds = tf.data.Dataset.from_generator(
            lambda: data_generator_full(val_low_folder, val_high_folder, image_size, batch_size),
            output_types=(tf.float32, tf.float32),
            output_shapes=((None, image_size[0], image_size[1], 3), (None, image_size[0], image_size[1], 3))
        )
    else:
        train_ds = tf.data.Dataset.from_generator(
            lambda: data_generator(low_folder, high_folder, crop_size, image_size, batch_size),
            output_types=(tf.float32, tf.float32),
            output_shapes=((None, crop_size, crop_size, 3), (None, crop_size, crop_size, 3))
        )

        val_ds = tf.data.Dataset.from_generator(
            lambda: data_generator(val_low_folder, val_high_folder, crop_size, image_size, batch_size),
            output_types=(tf.float32, tf.float32),
            output_shapes=((None, crop_size, crop_size, 3), (None, crop_size, crop_size, 3))
        )

    # Train the model
    model.fit(train_ds,
              steps_per_epoch=steps_per_epoch,
              epochs=epochs,
              callbacks=[lr_scheduler, checkpoint],
              validation_data=val_ds,
              validation_steps=validation_steps)

    return model

# Example usage
low_folder = '/content/vlg/Train/low'
high_folder = '/content/vlg/Train/high'
val_low_folder = '/content/vlg/validate/low'
val_high_folder = '/content/vlg/validate/high'

# Check if there are files matching the pattern
low_image_paths = glob.glob(os.path.join(low_folder, '*.png'))
if len(low_image_paths) == 0:
    raise ValueError("No PNG files found in the directory:", low_folder)

# Initial training on 256x256 crops
crop_size = 256
input_shape = (crop_size, crop_size, 3)

model = train_model(low_folder, high_folder, val_low_folder, val_high_folder, crop_size, input_shape, image_size=(400, 600))

# Save the initially trained model
model.save('denoising1_model.h5')

# Fine-tuning on full 400x600 images
input_shape_full = (400, 600, 3)
model_fine_tuned = train_model(low_folder, high_folder, val_low_folder, val_high_folder, crop_size, input_shape_full, image_size=(400, 600), fine_tune=True, model_path='denoising_initial_model.h5')

# Save the fine-tuned model
model_fine_tuned.save('denoising_fine_tuned_model.h5')