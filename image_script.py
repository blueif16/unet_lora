# resize_convert_script.py
import os
from PIL import Image
from tqdm import tqdm

import cv2
import numpy as np

def pad_image(image, target_size, fill):
    """
    Pads the given PIL Image to the target size.

    Args:
        image (PIL.Image): The image to pad.
        target_size (tuple): The desired size as (width, height).
        fill (int or tuple): Pixel fill value for padding.

    Returns:
        PIL.Image: The padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_size

    # Calculate padding
    pad_width = target_width - original_width
    pad_height = target_height - original_height

    if pad_width < 0 or pad_height < 0:
        raise ValueError("Target size must be larger than the original size.")

    # Calculate padding for each side
    padding = (
        pad_width // 2,                     # Left
        pad_height // 2,                    # Top
        pad_width - (pad_width // 2),       # Right
        pad_height - (pad_height // 2)      # Bottom
    )

    # Apply padding
    padded_image = Image.new(image.mode, target_size, fill)
    padded_image.paste(image, (padding[0], padding[1]))

    return padded_image

def resize_dataset(
    images_dir,
    masks_dir,
    output_images_dir,
    output_masks_dir,
    target_size=(1918, 1280),
    image_ext='.png',
    mask_ext='.png'
):
    """
    Resizes and pads images and masks to the target size.

    Args:
        images_dir (str): Path to the directory containing original images.
        masks_dir (str): Path to the directory containing original masks.
        output_images_dir (str): Path to save resized images.
        output_masks_dir (str): Path to save resized masks.
        target_size (tuple): Desired size as (width, height).
        image_ext (str): Image file extension.
        mask_ext (str): Mask file extension.
    """
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)

    # List all image files in images_dir with the specified extension
    image_files = [
        f for f in os.listdir(images_dir)
        if os.path.isfile(os.path.join(images_dir, f)) and f.lower().endswith(image_ext.lower())
    ]

    if not image_files:
        print(f"No images with extension '{image_ext}' found in {images_dir}.")
        return

    for img_file in tqdm(image_files, desc="Resizing Images and Masks"):
        img_path = os.path.join(images_dir, img_file)
        mask_path = os.path.join(masks_dir, img_file)  # Assuming same filename

        # Check if the corresponding mask exists
        if not os.path.exists(mask_path):
            print(f"Mask for image '{img_file}' not found in '{masks_dir}'. Skipping.")
            continue

        try:
            # Open image and mask
            image = Image.open(img_path).convert('RGB')  # Ensure image is in RGB
            mask = Image.open(mask_path).convert('RGB')    # Ensure mask is in grayscale

            # Pad image
            padded_image = pad_image(image, target_size, fill=(0, 0, 0))  # Black padding for images

            # Pad mask
            padded_mask = pad_image(mask, target_size, fill=(0, 0, 0))            # Zero padding for masks

            # Save resized images and masks as PNG
            resized_image_path_png = os.path.join(output_images_dir, img_file)
            resized_mask_path_png = os.path.join(output_masks_dir, img_file)  # Save mask with same filename

            padded_image.save(resized_image_path_png)
            padded_mask.save(resized_mask_path_png)

        except Exception as e:
            print(f"Error processing '{img_file}': {e}")

    print("Resizing completed successfully.")

def convert_png_to_jpg(
    input_dir,
    output_dir,
    file_ext='.png',
    quality=95
):
    """
    Converts PNG images to JPG format.

    Args:
        input_dir (str): Path to the directory containing PNG images.
        output_dir (str): Path to save converted JPG images.
        file_ext (str): File extension of input images (default: '.png').
        quality (int): Quality of the output JPG images (1-95).
    """
    os.makedirs(output_dir, exist_ok=True)

    # List all PNG files in input_dir
    png_files = [
        f for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(file_ext.lower())
    ]

    if not png_files:
        print(f"No images with extension '{file_ext}' found in {input_dir}.")
        return

    for png_file in tqdm(png_files, desc=f"Converting PNG to JPG in {input_dir}"):
        png_path = os.path.join(input_dir, png_file)
        jpg_filename = os.path.splitext(png_file)[0] + '.jpg'
        jpg_path = os.path.join(output_dir, jpg_filename)

        try:
            image = Image.open(png_path)
            # For masks, ensure they are in 'L' mode
            # if image.mode == 'L':
            #     # Optionally, convert to 'RGB' if necessary
            #     image = image.convert('RGB')
            # Save as JPG
            image.save(jpg_path, 'JPEG', quality=quality)
            os.remove(png_path)  # Remove original PNG file
        except Exception as e:
            print(f"Error converting '{png_file}' to JPG: {e}")

    print(f"Conversion to JPG completed successfully for directory: {input_dir}")

def car_classify(input_dir, output_dir, file_ext='.png'):
    os.makedirs(output_dir, exist_ok=True)

    # List all PNG files in input_dir
    png_files = [
        f for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(file_ext.lower())
    ]

    for png_file in tqdm(png_files, desc=f"Converting all color to black and white in {input_dir}"):
        png_path = os.path.join(input_dir, png_file)
        out_path = os.path.join(output_dir, png_file)

        # Load the image
        image = Image.open(png_path).convert('RGB')
        image_np = np.array(image)

        # Define the target color (white in RGB)
        target_color = np.array([0, 0, 142])
        # Create a boolean mask where pixels match the target color exactly
        match = np.all(image_np == target_color, axis=2)

        # Convert the boolean mask to an 8-bit binary mask
        binary_mask = np.zeros_like(image_np[:, :, 0], dtype=np.uint8)
        binary_mask[match] = 255

        # Convert to PIL Image and save
        binary_mask_image = Image.fromarray(binary_mask)
        binary_mask_image.save(out_path)

import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import logging

def convert_png_to_gif(
    input_dir,
    output_dir,
    file_ext='.png',
    threshold=128
):
    """
    Converts PNG images to GIF format, ensuring only black and white colors,
    and maintains the original image size.

    Args:
        input_dir (str): Path to the directory containing PNG images.
        output_dir (str): Path to save converted GIF images.
        file_ext (str): File extension of input images (default: '.png').
        threshold (int): Threshold for binarization (0-255). Pixels above this value are white, others are black.
    """
    # # Configure logging
    # logging.basicConfig(
    #     filename='gif_conversion.log',
    #     level=logging.INFO,
    #     format='%(asctime)s:%(levelname)s:%(message)s'
    # )

    os.makedirs(output_dir, exist_ok=True)

    # List all PNG files in input_dir
    png_files = [
        f for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(file_ext.lower())
    ]

    if not png_files:
        print(f"No images with extension '{file_ext}' found in {input_dir}.")
        return

    for png_file in tqdm(png_files, desc=f"Converting PNG to GIF in {input_dir}"):
        png_path = os.path.join(input_dir, png_file)
        gif_filename = os.path.splitext(png_file)[0] + '.gif'
        gif_path = os.path.join(output_dir, gif_filename)

        try:
            # Load the image and convert to grayscale ('L' mode)
            image = Image.open(png_path).convert('L')
            image_np = np.array(image)

            # Apply thresholding to binarize the image
            binary_mask = np.where(image_np > threshold, 1, 0).astype(np.uint8)

            # Convert the binary mask to '1' mode (binary image)
            binary_mask_image = Image.fromarray(binary_mask, mode='L').convert('1')

            print(binary_mask)
            return

            # Save the image as GIF with a fixed palette (black and white)
            binary_mask_image.save(gif_path, format='GIF')
            os.remove(png_path)  # Remove original PNG file

            # Optional: Remove the original PNG file after successful conversion
            # os.remove(png_path)

            logging.info(f"Successfully converted '{png_file}' to '{gif_filename}'.")
        except Exception as e:
            print(f"Error converting '{png_file}' to GIF: {e}")
            logging.error(f"Error converting '{png_file}' to GIF: {e}")

    print(f"Conversion to GIF completed successfully for directory: {input_dir}")

if __name__ == "__main__":
    # Define paths
    IMAGES_DIR = 'data/imgs/train_gta'                    # Replace with your actual images directory
    MASKS_DIR = 'data/masks/mask_gta'                      # Replace with your actual masks directory
    RESIZED_IMAGES_DIR = 'data/imgs/resized_images_gta'    # Directory to save resized images (PNG)
    RESIZED_MASKS_DIR = 'data/masks/resized_masks_gta'      # Directory to save resized masks (PNG)
    RESIZED_IMAGES_JPG_DIR = 'data/imgs/resized_images_gta'  # Directory to save resized images (JPG)
    RESIZED_MASKS_JPG_DIR = 'data/masks/resized_masks_gta'    # Directory to save resized masks (JPG)

    # Define target size
    TARGET_SIZE = (1918, 1280)                # (width, height)

    # Define file extensions
    IMAGE_EXT = '.png'                        # Images are in PNG format
    MASK_EXT = '.png'                         # Masks are in PNG format

    # Step 1: Resize and pad images and masks, save as PNG
    # print("Starting resizing and padding of images and masks...")
    # resize_dataset(
    #     images_dir=IMAGES_DIR,
    #     masks_dir=MASKS_DIR,
    #     output_images_dir=RESIZED_IMAGES_DIR,
    #     output_masks_dir=MASKS_DIR,
    #     target_size=TARGET_SIZE,
    #     image_ext=IMAGE_EXT,
    #     mask_ext=MASK_EXT
    # )

    # # Step 2: Convert resized PNG images to JPG
    # print("\nStarting conversion of resized images to JPG...")
    # convert_png_to_jpg(
    #     input_dir=RESIZED_IMAGES_DIR,
    #     output_dir=RESIZED_IMAGES_JPG_DIR,
    #     file_ext=IMAGE_EXT,
    #     quality=95  # Adjust quality as needed
    # )

    # car_classify(MASKS_DIR, RESIZED_MASKS_DIR)

    convert_png_to_gif(
    input_dir = RESIZED_MASKS_DIR,
    output_dir = RESIZED_MASKS_DIR)

    # # Step 3: Convert resized PNG masks to JPG
    # print("\nStarting conversion of resized masks to JPG...")
    # convert_png_to_jpg(
    #     input_dir=RESIZED_MASKS_DIR,
    #     output_dir=RESIZED_MASKS_JPG_DIR,
    #     file_ext=MASK_EXT,
    #     quality=95  # Adjust quality as needed
    # )



    print("\nAll processes completed successfully.")
