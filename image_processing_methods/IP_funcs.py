import numpy as np
import cv2
import rdp


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

    frequency = np.random.uniform(0.5, 2.0)  # Random frequency multiplier
    phase_shift = np.random.uniform(0, np.pi)  # Random phase shift
    x_direction = np.random.choice([-1, 1])  # Random direction in x-axis
    y_direction = np.random.choice([-1, 1])  # Random direction in y-axis

    x_indices, y_indices = np.meshgrid(np.arange(width), np.arange(height))

    if non_linear:
        base_value = visibility * (
            1 + np.sin(
                frequency * (x_direction * y_indices + y_direction * x_indices) / (height + width) * np.pi + phase_shift
            )
        )
    else:
        base_value = visibility * (y_indices + x_indices) / (height + width)

    # Add random variations
    random_variation = np.random.uniform(-randomness, randomness, (height, width))
    alpha = 1 - base_value + random_variation

    return np.clip(alpha, 0.25, 1)


def customAddWeighted(src1, alpha, src2, beta, gamma=0):
    # Check if the images have the same size
    if src1.shape != src2.shape:
        raise ValueError("Input images must have the same size.")

    # Perform alpha blending
    blended_image = np.clip(src1 * alpha[:, :, np.newaxis] + src2 * beta[:, :, np.newaxis] + gamma, 0, 255).astype(
        np.uint8)

    return blended_image


def apply_alpha_blending(rgb_image, soil_image):
    alpha = generate_random_alpha_gradient(rgb_image.shape[:2], non_linear=True)

    blended_image = customAddWeighted(rgb_image, alpha, soil_image, 1-alpha, 0)

    return blended_image


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



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Define image size and gradient parameters
    img_size = (200, 200)
    visibility = 0.8
    randomness = 0.02

    # Generate a random non-linear alpha gradient
    alpha_gradient = generate_random_alpha_gradient(img_size, non_linear=True)

    # Plot the generated alpha gradient
    plt.imshow(alpha_gradient, cmap='gray')
    plt.colorbar(label='Alpha Value')
    plt.title('Non-linear Alpha Gradient Effect')
    plt.show()

    img_size = (200, 200, 3)
    image = np.full(img_size, 150, dtype=np.uint8)  # Gray image

    # Apply the light effect to the image
    intensity = 1.5
    spread = 0.4
    light_effect_image = add_light_effect(image, intensity=intensity, spread=spread)

    # Plot the original and light-effect images side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(light_effect_image)
    axes[1].set_title('Image with Light Effect')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


