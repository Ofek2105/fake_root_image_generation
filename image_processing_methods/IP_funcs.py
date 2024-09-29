import numpy as np
import cv2
import rdp
from PIL import Image
import io
import Augmentor

def get_polygons_bbox_from_bin_image(bin_image, neg_mask=None):
    result = bin_image.copy()
    if neg_mask is not None:
        result = np.logical_and(bin_image.copy(), np.logical_not(neg_mask.copy()))

    # Dilate the binary image to thicken thin objects
    kernel = np.ones((3, 3), np.uint8)
    dilated_result = cv2.dilate(result.astype(np.uint8), kernel, iterations=1)

    # Find contours using OpenCV
    contours, _ = cv2.findContours(dilated_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        contour_points = np.squeeze(max_contour).astype(np.float32)

        if contour_points.ndim == 1:  # Handle the case where contour is a single point
            contour_points = np.expand_dims(contour_points, axis=0)

        contour_points[:, 0] = contour_points[:, 0] / bin_image.shape[0]
        contour_points[:, 1] = contour_points[:, 1] / bin_image.shape[1]

        # Simplify the contour using RDP
        keep = rdp.rdp(contour_points.tolist(), epsilon=1e-3, algo="iter", return_mask=True)
        polygon_points = contour_points[keep].flatten().tolist()

        # Calculate bounding box
        min_x, min_y = np.min(contour_points, axis=0)
        max_x, max_y = np.max(contour_points, axis=0)
        bbox = [min_x, min_y, max_x - min_x, max_y - min_y]

        return polygon_points, bbox
    return [[]], []


def generate_random_alpha_gradient(img_size, non_linear=True, visibility=0.8, randomness=0.01):
    height, width = img_size
    alpha = np.zeros((height, width), dtype=np.float32)

    # Randomize direction, frequency, and phase
    frequency = np.random.uniform(0.5, 2.0)  # Random frequency multiplier
    phase_shift = np.random.uniform(0, np.pi)  # Random phase shift
    x_direction = np.random.choice([-1, 1])  # Random direction in x-axis
    y_direction = np.random.choice([-1, 1])  # Random direction in y-axis

    # Generate a random gradient with opacity variations
    for i in range(height):
        for j in range(width):
            if non_linear:
                # Apply random frequency, phase, and direction
                base_value = visibility * (
                        1 + np.sin(
                    frequency * (x_direction * i + y_direction * j) / (height + width) * np.pi + phase_shift)
                )
            else:
                base_value = visibility * (i + j) / (height + width)

            # Add some random variations for a more natural effect
            random_variation = np.random.uniform(-randomness, randomness)
            alpha[i, j] = 1 - base_value + random_variation

    return np.clip(alpha, 0.25, 1)


def apply_alpha_blending(bin_image, soil_image, alpha):
    # Ensure both images are of the same size
    if bin_image.shape[:2] != soil_image.shape[:2]:
        raise ValueError("Binary image and soil image must be the same size.")

    bin_mask = bin_image == 1
    soil_image[bin_mask, :] = (alpha[bin_mask][..., np.newaxis] * soil_image[bin_mask, :] +
                               (1 - alpha[bin_mask])[..., np.newaxis] * np.array([255, 255, 255])[np.newaxis, ...]).astype(np.uint8)

    return soil_image


def add_light_effect(image, intensity=1.5, spread=0.4):
    height, width, channels = image.shape

    gradient = np.linspace(1, 0, height).reshape(height, 1)
    gradient = np.clip(gradient + spread, 0, 1)
    mask = np.repeat(gradient, width, axis=1)
    mask = np.repeat(mask[:, :, np.newaxis], channels, axis=2)

    image = image.astype(np.float32)
    mask = mask.astype(np.float32)

    brightened_image = cv2.addWeighted(image, 1.0, image * mask * intensity, 0.8, 0)

    return np.clip(brightened_image, 0, 255).astype(np.uint8)
