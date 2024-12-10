import numpy as np
import cv2
import rdp
import random


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
        contour_points = np.squeeze(max_contour)

        if contour_points.ndim == 1:  # Handle the case where contour is a single point
            contour_points = np.expand_dims(contour_points, axis=0)

        # contour_points[:, 0] = contour_points[:, 0] / bin_image.shape[0]
        # contour_points[:, 1] = contour_points[:, 1] / bin_image.shape[1]

        # Simplify the contour using RDP
        keep = rdp.rdp(contour_points.tolist(), epsilon=1, algo="iter", return_mask=True)
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

    frequency = np.random.uniform(0.5, 1)  # Random frequency multiplier
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

    blended_image = customAddWeighted(rgb_image, alpha, soil_image, 1 - alpha, 0)

    return blended_image


def back_for_ground_blending(rgb_image, soil_image):
    if rgb_image.shape != soil_image.shape:
        raise ValueError("Foreground and background images must have the same dimensions.")

    gray_foreground = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_foreground, 1, 255, cv2.THRESH_BINARY)
    mask = cv2.GaussianBlur(mask, (21, 21), 10) / 255.0
    mask = np.stack([mask] * 3, axis=-1)
    blended = (rgb_image * mask + soil_image * (1 - mask)).astype(np.uint8)
    return blended


def add_channel_noise(image, stddev=5, apply_chane=1.0):
    if random.random() > apply_chane:
        return image

    noise = np.random.normal(0, stddev, image.shape).astype(np.float32)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image


# def add_light_effect(image, intensity=1.5, spread=0.4, apply_chance=0.5):
#
#     if random.random() > apply_chance:
#         return image
#
#     height, width, channels = image.shape
#
#     gradient = np.linspace(1, 0, height).reshape(height, 1)
#     gradient = np.clip(gradient + spread, 0, 1)
#     mask = np.repeat(gradient, width, axis=1)
#     mask = np.repeat(mask[:, :, np.newaxis], channels, axis=2)
#
#     image = image.astype(np.float32)
#     mask = mask.astype(np.float32)
#
#     brightened_image = cv2.addWeighted(image, 1.0, image * mask * intensity, 0.8, 0)
#
#     return np.clip(brightened_image, 0, 255).astype(np.uint8)


def add_light_effect(image, min_intensity=0.2, max_intensity=1.8,
                     min_freq=0.2, max_freq=2, apply_chance=0.5):
    """
    Apply a random directional gradient effect to an image.

    Args:
        image: Input image (numpy array)
        min_intensity: Minimum brightness intensity (default: 0.2)
        max_intensity: Maximum brightness intensity (default: 0.8)
        min_freq: Minimum frequency multiplier (default: 1)
        max_freq: Maximum frequency multiplier (default: 3)
        apply_chance: Probability of applying the effect (default: 0.5)

    Returns:
        Modified image with gradient effect
    """
    if np.random.random() > apply_chance:
        return image

    height, width, channels = image.shape

    # Random parameters
    angle = np.random.uniform(0, 360)
    intensity = np.random.uniform(min_intensity, max_intensity)
    freq = np.random.uniform(min_freq, max_freq)

    # Create basic gradient along y-axis
    gradient = np.linspace(0, 1, height) * freq
    gradient = gradient.reshape(height, 1)

    # Create coordinate grid
    y, x = np.mgrid[0:height, 0:width]

    # Convert angle to radians
    theta = np.radians(angle)

    # Rotate coordinates
    x_rot = x * np.cos(theta) + y * np.sin(theta)
    y_rot = -x * np.sin(theta) + y * np.cos(theta)

    # Normalize coordinates to [0, 1]
    y_rot = (y_rot - y_rot.min()) / (y_rot.max() - y_rot.min())

    # Create rotated gradient
    mask = y_rot
    mask = np.clip(mask, 0, 1)
    mask = np.repeat(mask[:, :, np.newaxis], channels, axis=2)

    # Apply gradient
    image = image.astype(np.float32)
    mask = mask.astype(np.float32)

    brightened = cv2.addWeighted(
        image, 1.0,
        image * mask * intensity, 1,
        0
    )

    return (brightened / np.max(brightened) * 255).astype(np.uint8)
    return np.clip(brightened, 0, 255).astype(np.uint8)


def apply_motion_blur(image, degree=5, apply_chane=0.5):
    if random.random() > apply_chane:
        return image

    # Create a motion blur kernel
    M = np.zeros((degree, degree))
    M[int((degree - 1) / 2), :] = np.ones(degree)

    angle = np.random.uniform(-180, 180)

    M = cv2.warpAffine(M, cv2.getRotationMatrix2D((degree / 2 - 0.5, degree / 2 - 0.5), angle, 1.0), (degree, degree))
    M = M / degree

    # Apply the kernel to the image
    blurred = cv2.filter2D(image, -1, M)
    return blurred


def apply_random_vortex_blur(img, strength=0.0005, apply_chane=0.5):
    if random.random() > apply_chane:
        return img

    h, w, _ = img.shape
    cx = np.random.randint(w // 4, 3 * w // 4)
    cy = np.random.randint(h // 4, 3 * h // 4)
    y, x = np.indices((h, w))
    dx = x - cx
    dy = y - cy
    distance = np.sqrt(dx ** 2 + dy ** 2)
    angle = np.arctan2(dy, dx)
    angle_distorted = angle + strength * distance
    new_x = cx + distance * np.cos(angle_distorted)
    new_y = cy + distance * np.sin(angle_distorted)
    new_x = np.clip(new_x, 0, w - 1).astype(np.float32)
    new_y = np.clip(new_y, 0, h - 1).astype(np.float32)
    vortex_blurred = np.zeros_like(img)
    for i in range(3):
        vortex_blurred[..., i] = cv2.remap(img[..., i], new_x, new_y, interpolation=cv2.INTER_LINEAR)
    return vortex_blurred


def apply_gaussian_blurr(img, apply_chane=0.5):
    if random.random() > apply_chane:
        return img

    blur_strength = np.random.choice([5, 7, 11, 11, 15, 31])
    print(blur_strength)
    return cv2.GaussianBlur(img, (blur_strength, blur_strength), 0)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    for _ in range(30):
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
        light_effect_image = add_light_effect(image, apply_chance=1)

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
