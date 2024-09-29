import cv2
import os
import numpy as np
import random


def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    return images


# Function to resize an image to the desired size
def resize_image(img, size):
    return cv2.resize(img, size)


# Function to randomly rotate an image
def rotate_image(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)


# Function to generate a random polygon mask
def apply_polygon_mask(patch):
    h, w = patch.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Define random polygon points (minimum 3 points)
    num_points = random.randint(3, 10)
    points = np.array([[
        (random.randint(0, w), random.randint(0, h))
        for _ in range(num_points)
    ]], dtype=np.int32)

    # Create polygon mask
    cv2.fillPoly(mask, points, (255))

    # Feathering (optional, for smoother borders)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)

    return mask


# Function to randomly extract a patch from an image
def extract_random_patch(img, patch_size):
    h, w = img.shape[:2]
    ph, pw = patch_size

    # Ensure patch size is smaller than the image
    if ph > h or pw > w:
        ph = min(ph, h)
        pw = min(pw, w)

    x = random.randint(0, w - pw)
    y = random.randint(0, h - ph)
    patch = img[y:y + ph, x:x + pw]

    # Apply random rotation
    angle = random.uniform(0, 360)
    patch = rotate_image(patch, angle)

    # Apply a random polygon mask
    mask = apply_polygon_mask(patch)

    return patch, mask


# Function to blend patches with alpha blending
def alpha_blend_patches(base_image, patch, mask, x, y):
    ph, pw = patch.shape[:2]

    # Convert mask to float and normalize to range [0, 1]
    mask = mask.astype(np.float32) / 255.0

    # Define the region of interest (ROI) in the base image
    roi = base_image[y:y + ph, x:x + pw]

    # Apply alpha blending: patch * alpha + base_image * (1 - alpha)
    blended = (patch * mask[..., np.newaxis] + roi * (1 - mask[..., np.newaxis])).astype(np.uint8)

    # Place the blended region back into the base image
    base_image[y:y + ph, x:x + pw] = blended


# Function to construct the final image
def construct_image(images, image_size, max_patches=10, min_patch_size=(400, 400),
                    max_patch_size=(500, 500)):
    # Start with a base image resized to the full screen
    base_image = resize_image(random.choice(images), image_size)

    # Randomly select up to max_images images for patches
    selected_images = random.sample(images, k=len(images))

    for _ in range(max_patches):
        img = random.choice(selected_images)

        # Random patch size
        patch_width = random.randint(min_patch_size[0], min(max_patch_size[0], image_size[0]))
        patch_height = random.randint(min_patch_size[1], min(max_patch_size[1], image_size[1]))

        # Extract random patch from the image and its corresponding mask
        patch_img = resize_image(img, image_size)
        patch, mask = extract_random_patch(patch_img, (patch_height, patch_width))

        # Random position to place the patch, ensure it fits within the base image
        x = random.randint(0, image_size[0] - patch_width)
        y = random.randint(0, image_size[1] - patch_height)

        # Alpha blend the patch onto the base image
        alpha_blend_patches(base_image, patch, mask, x, y)

    return base_image


def generate_and_save(images, output_size, output_folder="generated_backgrounds", num_generated=10, max_images=3):

    for i in range(num_generated):
        new_image = construct_image(images, output_size)
        output_path = os.path.join(output_folder, f"generated_soil_{i + 1}.png")
        cv2.imwrite(output_path, new_image)


def gen_soil_image(output_size, soil_images_folder_path='background_resources'):
    soil_images = load_images(soil_images_folder_path)
    return construct_image(soil_images, output_size)


def soil_image_generator(N, output_size, soil_images_folder_path='background_resources'):
    soil_images = load_images(soil_images_folder_path)
    for _ in range(N):
        yield construct_image(soil_images, output_size)


if __name__ == "__main__":
    image_folder = "background_resources"
    soil_images = load_images(image_folder)

    generate_and_save(soil_images, output_size=(960, 960), num_generated=10, max_images=5)
