import cv2
import numpy as np
import os
from dataclasses import dataclass
import random
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, List, Generator


@dataclass
class SoilPatch:
    image: np.ndarray
    mask: np.ndarray
    position: Tuple[int, int]
    size: Tuple[int, int]

class SoilGenerator:
    def __init__(self, image_folder: str = ""):
        """Initialize the soil generator with a folder of soil texture images."""
        self.images = self._load_images(image_folder)
        if not self.images:
            raise ValueError(f"No valid images found in {image_folder}")

        # Pre-calculate average soil color for reference
        self.reference_color = self._calculate_reference_color()

    def _load_images(self, folder: str) -> List[np.ndarray]:
        """Load and preprocess soil texture images."""
        images = []
        folder_path = Path(folder)

        for img_path in folder_path.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                # Read in BGR and convert to RGB
                img = cv2.imread(str(img_path))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # Adjust contrast and brightness
                    img = self._enhance_texture(img)
                    images.append(img)

        return images

    def _check_spacing(self, new_patch: SoilPatch, existing_patches: List[SoilPatch], min_distance: int) -> bool:
        """
        Check if a new patch maintains minimum distance from existing patches.
        Returns True if spacing is acceptable, False otherwise.
        """
        new_x, new_y = new_patch.position
        new_h, new_w = new_patch.size
        new_center = (new_x + new_w // 2, new_y + new_h // 2)

        for patch in existing_patches:
            x, y = patch.position
            h, w = patch.size
            center = (x + w // 2, y + h // 2)

            # Calculate distance between centers
            distance = np.sqrt((new_center[0] - center[0]) ** 2 + (new_center[1] - center[1]) ** 2)

            # Minimum required distance based on patch sizes
            required_distance = min_distance + (np.sqrt(new_w ** 2 + new_h ** 2) +
                                                np.sqrt(w ** 2 + h ** 2)) / 4

            if distance < required_distance:
                return False
        return True

    def _find_valid_position(self, patch_size: Tuple[int, int], image_size: Tuple[int, int],
                             existing_patches: List[SoilPatch], min_distance: int,
                             max_attempts: int = 50) -> Tuple[int, int]:
        """Find a valid position for a new patch that maintains minimum spacing."""
        ph, pw = patch_size
        img_h, img_w = image_size

        for _ in range(max_attempts):
            # Random position with margin
            x = random.randint(0, img_w - pw)
            y = random.randint(0, img_h - ph)

            # Create temporary patch for spacing check
            temp_patch = SoilPatch(
                image=np.zeros((ph, pw, 3)),  # Dummy image
                mask=np.zeros((ph, pw)),  # Dummy mask
                position=(x, y),
                size=(ph, pw)
            )

            if self._check_spacing(temp_patch, existing_patches, min_distance):
                return x, y

        # If no valid position found after max attempts, return None
        return None

    def _enhance_texture(self, img: np.ndarray) -> np.ndarray:
        """Enhance soil texture while preserving natural colors."""
        # Convert to LAB for selective contrast enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to luminance only
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Merge back and convert to RGB
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def _calculate_reference_color(self) -> np.ndarray:
        """Calculate average soil color from all images."""
        all_colors = []
        for img in self.images:
            # Sample random points from each image
            h, w = img.shape[:2]
            points = np.random.choice(h * w, 1000)
            colors = img.reshape(-1, 3)[points]
            all_colors.extend(colors)

        return np.median(all_colors, axis=0)

    def _generate_random_polygon_points(self, center: np.ndarray, min_radius: float, max_radius: float,
                                        num_vertices: int) -> np.ndarray:
        """Generate random points for an irregular polygon."""
        angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
        angles += np.random.uniform(-0.2, 0.2, num_vertices)  # Add some randomness to angles

        # Generate random radii with smooth transitions
        radii = []
        base_radius = random.uniform(min_radius, max_radius)
        prev_radius = base_radius

        for _ in range(num_vertices):
            # Create smooth transition between radii
            radius = prev_radius + random.uniform(-0.2, 0.2) * base_radius
            radius = max(min_radius, min(max_radius, radius))
            radii.append(radius)
            prev_radius = radius

        # Generate points
        points = []
        for angle, radius in zip(angles, radii):
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            points.append([int(x), int(y)])

        return np.array(points)

    def _create_natural_patch_mask(self, size: Tuple[int, int]) -> np.ndarray:
        """Create an organic-looking patch mask with random polygon shapes."""
        height, width = size
        mask = np.zeros((height, width), dtype=np.float32)
        center = np.array([width / 2, height / 2])

        # Create main polygon
        min_radius = min(width, height) * 0.2
        max_radius = min(width, height) * 0.55
        num_vertices = random.randint(6, 25)  # More vertices for more variation

        main_points = self._generate_random_polygon_points(center, min_radius, max_radius, num_vertices)

        # Add sub-polygons for more organic look
        num_sub_polygons = random.randint(2, 4)
        all_polygons = [main_points]

        for _ in range(num_sub_polygons):
            # Generate random center point near main polygon
            offset = np.random.uniform(-0.3, 0.3, 2) * min(width, height)
            sub_center = center + offset

            # Create smaller polygon
            sub_min_radius = min_radius * 0.4
            sub_max_radius = max_radius * 0.4
            sub_vertices = random.randint(5, 8)

            sub_points = self._generate_random_polygon_points(sub_center, sub_min_radius, sub_max_radius, sub_vertices)
            all_polygons.append(sub_points)

        # Draw and fill all polygons
        cv2.fillPoly(mask, [np.int32(poly) for poly in all_polygons], 1.0)

        # Apply multiple gaussian blurs with different kernel sizes for more natural edges
        mask = cv2.GaussianBlur(mask, (5, 5), sigmaX=10)
        mask = cv2.GaussianBlur(mask, (21, 21), sigmaX=100)

        return mask

    def _extract_patch(self, img: np.ndarray, size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract a random patch from image with natural mask."""
        h, w = img.shape[:2]
        ph, pw = size

        # Ensure patch size fits within image
        ph = min(ph, h)
        pw = min(pw, w)

        # Random position
        x = random.randint(0, w - pw)
        y = random.randint(0, h - ph)

        # Extract patch and create mask
        patch = img[y:y + ph, x:x + pw].copy()
        mask = self._create_natural_patch_mask((ph, pw))

        return patch, mask

    def _blend_patch(self, base: np.ndarray, patch: SoilPatch) -> np.ndarray:
        """Blend patch into base image with natural transition."""
        x, y = patch.position
        ph, pw = patch.image.shape[:2]

        # Extract ROI
        roi = base[y:y + ph, x:x + pw]

        # Prepare mask for blending
        mask_3ch = np.stack([patch.mask] * 3, axis=-1)

        # Blend images
        blended = (patch.image * mask_3ch + roi * (1 - mask_3ch)).astype(np.uint8)

        # Copy result back to base image
        base[y:y + ph, x:x + pw] = blended
        return base

    def generate_image(self, size: Tuple[int, int], num_patches: int = 10,
                       min_distance: int = 40) -> np.ndarray:
        """
        Generate a realistic soil background image with spaced patches.

        Args:
            size: Tuple of (height, width) for the output image
            num_patches: Number of patches to generate
            min_distance: Minimum distance between patch centers
        """
        # Start with a base texture
        base_img = cv2.resize(random.choice(self.images), size)
        existing_patches = []

        # Generate and apply patches
        attempts = 0
        max_attempts = num_patches * 15  # Maximum attempts to place all patches

        while len(existing_patches) < num_patches and attempts < max_attempts:
            # Random patch size
            patch_size = (
                random.randint(size[0] // 5, size[0] // 2),  # Slightly smaller patches
                random.randint(size[1] // 5, size[1] // 2)
            )

            # Find valid position
            position = self._find_valid_position(patch_size, size, existing_patches, min_distance)

            if position is not None:
                x, y = position

                # Extract patch from random image
                source_img = random.choice(self.images)
                patch_img, mask = self._extract_patch(source_img, patch_size)

                # Create and store patch
                patch = SoilPatch(patch_img, mask, (x, y), patch_size)
                existing_patches.append(patch)

                # Blend patch into base image
                base_img = self._blend_patch(base_img, patch)

            attempts += 1

        return base_img

    def generate_multiple(self, size: Tuple[int, int], count: int,
                          min_distance: int = 100) -> Generator[np.ndarray, None, None]:
        """Generate multiple soil background images with spaced patches."""
        for _ in range(count):
            yield self.generate_image(size, min_distance=min_distance)


def plot_soil_image(img: np.ndarray):
    """Utility function to display the generated soil image."""
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()


# Example usage:
if __name__ == "__main__":
    # Initialize generator
    generator = SoilGenerator("background_resources")

    # Generate a single image
    soil_img = generator.generate_image((960, 960))
    plot_soil_image(soil_img)

    # Generate multiple images
    for img in generator.generate_multiple((960, 960), 5):
        plt.imshow(img)
        plt.show()