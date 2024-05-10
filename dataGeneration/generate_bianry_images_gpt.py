import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from skimage import draw


def generate_hairy_image_v4(size=(800, 1200), body_width=20, body_length=70,
                            max_hair_length=50, min_hair_length=30,
                            hair_thickness=2, n_hairs=300):
  # Create an empty image
  image = np.zeros(size, dtype=np.uint8)

  # Initialize the main body starting point
  x, y = size[1] // 2, size[0] // 2

  # Generate a random walk for the main body with a more controlled angle change
  direction = np.random.uniform(-np.pi, np.pi)
  steps = size[0] // body_length * 2
  for _ in range(steps):
    # Reduced angle deviation for less drastic changes in direction
    angle = direction + np.random.uniform(-np.pi / 8, np.pi / 8)
    x_end = np.clip(int(x + np.cos(angle) * body_length), 0, size[1] - 1)
    y_end = np.clip(int(y + np.sin(angle) * body_length), 0, size[0] - 1)
    rr, cc = draw.line(y, x, y_end, x_end)
    image[rr, cc] = 1
    x, y = x_end, y_end

  # Dilate the main body to make it thicker
  image = morphology.binary_dilation(image, morphology.disk(body_width // 2))

  # Add hairs with random lengths and directions
  for _ in range(n_hairs):
    # Choose a random point on the main body
    where_body = np.where(image.ravel() == 1)[0]
    if len(where_body) == 0:  # If the main body is not in the image, skip hair generation
      continue
    y, x = np.random.choice(where_body, size=2, replace=False)
    y, x = y // size[1], y % size[1]
    angle = np.random.uniform(0, 2 * np.pi)
    hair_length = np.random.randint(min_hair_length, max_hair_length)
    x_end = np.clip(int(x + np.cos(angle) * hair_length), 0, size[1] - 1)
    y_end = np.clip(int(y + np.sin(angle) * hair_length), 0, size[0] - 1)
    rr, cc = draw.line(y, x, y_end, x_end)
    image[rr, cc] = 1

  # Dilate the hairs to make them thicker
  image = morphology.binary_dilation(image, morphology.disk(hair_thickness))

  return image


# Generate and display three binary images with the new specifications
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax in axes:
  binary_image_v4 = generate_hairy_image_v4()
  ax.imshow(binary_image_v4, cmap='gray')
  ax.axis('off')

plt.tight_layout()
plt.show()