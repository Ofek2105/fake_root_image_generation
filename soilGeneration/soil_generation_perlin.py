import matplotlib.pyplot as plt
import numpy as np
from noise import pnoise2
from matplotlib.colors import LinearSegmentedColormap
import random


class SoilTextureGenerator:
    def generate_perlin_noise(self, width, height, scale=10, octaves=6, persistence=0.5, lacunarity=2.0, seed=None):
        if seed is not None:
            np.random.seed(seed)
        noise_map = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                noise_map[i][j] = pnoise2(
                    j / scale, i / scale, octaves=octaves,
                    persistence=persistence, lacunarity=lacunarity,
                    repeatx=width, repeaty=height
                )
        return (noise_map - noise_map.min()) / (noise_map.max() - noise_map.min())

    def add_granular_noise(self, image, intensity=0.1):
        granular_noise = np.random.uniform(0, intensity, image.shape)
        return np.clip(image + granular_noise, 0, 1)

    def threshold_clumps(self, image, thresholds):
        clumped = np.zeros_like(image)
        for i, t in enumerate(thresholds):
            clumped[image > t] = i / len(thresholds)
        return clumped

    def add_streaks(self, image, direction="horizontal", intensity=0.1):
        streak = np.linspace(0, intensity, image.shape[1] if direction == "horizontal" else image.shape[0])
        streak = streak if direction == "horizontal" else streak[:, None]
        return np.clip(image + streak, 0, 1)

    def add_shadows_highlights(self, image, shadow_intensity=0.2, highlight_intensity=0.2):
        shadow = np.sin(np.linspace(0, np.pi, image.shape[1])) * shadow_intensity
        shadow = shadow[:, None]
        highlight = np.cos(np.linspace(0, np.pi, image.shape[0])) * highlight_intensity
        highlight = highlight[None, :]
        return np.clip(image + shadow + highlight, 0, 1)

    def blend_textures(self, texture1, texture2, alpha=0.5):
        return np.clip(alpha * texture1 + (1 - alpha) * texture2, 0, 1)

    def apply_soil_color_palette(self, image):
        def normalize_colors(colors):
            return [(r / 255, g / 255, b / 255) for r, g, b in colors]

        # Extended soil color palette
        extended_soil_colors = [
            (102, 51, 0),  # Dark earthy brown
            (153, 101, 21),  # Dark brown
            (186, 140, 99),  # Medium brown
            (210, 180, 140),  # Light brown
            (255, 219, 180),  # Sandy brown
            (139, 69, 19),  # Saddle brown
            (205, 133, 63),  # Peru
            (244, 164, 96),  # Sandy orange
            (112, 128, 144),  # Slate gray
            (169, 169, 169)  # Dim gray
        ]

        # Add randomness to the palette
        random.shuffle(extended_soil_colors)
        selected_colors = random.sample(extended_soil_colors, k=random.randint(3, len(extended_soil_colors)))

        normalized_colors = normalize_colors(selected_colors)
        colormap = LinearSegmentedColormap.from_list("RandomSoil", normalized_colors)
        colored_image = colormap(image)

        return (colored_image[..., :3] * 255).astype(np.uint8), colormap

    def generate_soil_texture(self, width=512, height=512):
        scale = random.uniform(50, 100)  # Randomize scale
        octaves = random.randint(4, 6)  # Randomize octaves
        persistence = random.uniform(0.7, 0.9)  # Randomize persistence
        lacunarity = random.uniform(1.5, 2.0)  # Randomize lacunarity
        intensity = random.uniform(0.1, 0.2)  # Randomize granular noise intensity

        image = self.generate_perlin_noise(width, height, scale=scale, octaves=octaves, persistence=persistence,
                                           lacunarity=lacunarity)
        image = self.add_granular_noise(image, intensity=intensity)
        clump_thresholds = [random.uniform(0.2, 0.4), random.uniform(0.4, 0.5), random.uniform(0.6, 0.7)]
        image = self.threshold_clumps(image, thresholds=clump_thresholds)
        streak_intensity = random.uniform(0.1, 0.3)
        image = self.add_streaks(image, direction="horizontal", intensity=streak_intensity)
        shadow_intensity = random.uniform(0.1, 0.3)
        highlight_intensity = random.uniform(0.1, 0.3)
        image = self.add_shadows_highlights(image, shadow_intensity=shadow_intensity,
                                            highlight_intensity=highlight_intensity)
        fine_texture = self.generate_perlin_noise(width, height, scale=random.uniform(10, 30),
                                                  octaves=random.randint(6, 12))
        image = self.blend_textures(image, fine_texture, alpha=random.uniform(0.5, 0.8))
        return self.apply_soil_color_palette(image / np.max(image))


if __name__ == "__main__":
    # Display Generated Soil Textures
    width, height = 512, 512

    for _ in range(10):  # Generate 10 textures
        soil_texture, colormap = SoilTextureGenerator().generate_soil_texture(width, height)
        a = plt.imshow(soil_texture)
        plt.axis('off')
        plt.title("Generated Soil Texture")
        plt.colorbar(
            plt.cm.ScalarMappable(cmap=colormap),
            ax=plt.gca(),
            orientation='vertical',
            fraction=0.03,
            pad=0.04)
        plt.show()
