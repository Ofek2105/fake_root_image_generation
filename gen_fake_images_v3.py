import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import bezier_curve
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import binary_dilation, generate_binary_structure


class RootImageGenerator:
  def __init__(self, points, root_width=10, epsilon_std=1, hair_length=50, hair_thickness=5, hair_n=10, img_width=1000,
               img_height=1000):
    self.points = points
    self.root_width = root_width
    self.epsilon_std = epsilon_std
    self.hair_length = hair_length
    self.hair_thickness = hair_thickness
    self.hair_n = hair_n
    self.img_width = img_width
    self.img_height = img_height
    self.normals = self.compute_normals()

  def compute_normals(self):
    # Vectorized normal calculation
    deltas = np.diff(self.points, axis=0)
    norms = np.linalg.norm(deltas, axis=1, keepdims=True)
    normals = np.c_[deltas[:, 1], -deltas[:, 0]] / norms
    return np.vstack([normals, normals[-1]])

  def generate_parallel_lines(self):
    offsets = np.random.normal(0, self.epsilon_std, size=(len(self.points), 1))
    line1 = self.points + self.normals * (self.root_width + offsets)
    line2 = self.points - self.normals * (self.root_width + offsets)
    return np.clip(line1, 0, [self.img_width - 1, self.img_height - 1]).astype(int), np.clip(line2, 0,
                                                                                             [self.img_width - 1,
                                                                                              self.img_height - 1]).astype(
      int)

  def draw_connections(self, bin_img, lin1, lin2):
    # Combined lin1 and lin2 to draw the outline of the root
    for start, end in zip(np.vstack([lin1, lin1[0]]), np.vstack([lin2, lin2[0]])):
      rr, cc = bezier_curve(start[1], start[0], (start[1] + end[1]) // 2, (start[0] + end[0]) // 2, end[1], end[0], 1)
      bin_img[rr, cc] = 1

  def draw_hairs(self, bin_image):
    length_variations = np.random.normal(0, self.hair_length * 0.2, size=self.hair_n)
    thickness_variations = np.random.normal(0, self.hair_thickness * 0.2, size=self.hair_n)

    sampled_indices = np.linspace(0, len(self.points) - 1, self.hair_n).astype(int)
    for i, point_idx in enumerate(sampled_indices):
      point = np.clip(self.points[point_idx], 0, [self.img_width - 1, self.img_height - 1]).astype(int)
      normal = self.normals[point_idx]

      if np.random.rand() > 0.5:
        normal = -normal

      length = max(5, self.hair_length + length_variations[i])
      thickness = max(1, self.hair_thickness + thickness_variations[i])
      control_point = (point + normal * length / 2 + np.random.normal(0, length / 4, size=2)).astype(int)
      end_point = (point + normal * length).astype(int)

      rr, cc = bezier_curve(point[1], point[0], control_point[1], control_point[0], end_point[1], end_point[0], 1)
      valid = (rr >= 0) & (rr < self.img_height) & (cc >= 0) & (cc < self.img_width)
      temp_image = np.zeros_like(bin_image)
      temp_image[rr[valid], cc[valid]] = 1

      # Apply dilation here to increase thickness
      struct = generate_binary_structure(2, 1)  # You can experiment with different structures
      temp_image = binary_dilation(temp_image, structure=struct, iterations=int(thickness // 2))

      bin_image |= temp_image

  def generate(self):
    # Generate parallel lines for the root structure
    line1, line2 = self.generate_parallel_lines()
    binary_image = np.zeros((self.img_height, self.img_width), dtype=np.uint8)

    # Draw connections to create the root shape
    self.draw_connections(binary_image, line1, line2)

    # Fill the root shape
    binary_image = binary_fill_holes(binary_image)

    # Draw hairs on the filled root
    self.draw_hairs(binary_image)

    return binary_image


# Generate points for the root shape
x = np.linspace(100, 400, 500)  # More points for a smoother curve
y = 200 + np.sin(x / 50) * 20 + np.sin(x / 100) * 40  # Combination of sine waves
points = np.vstack((x, y)).T

# Initialize the RootImageGenerator with the points and other parameters
root_image_generator = RootImageGenerator(points)

# Generate the binary image using the generate method
binary_image = root_image_generator.generate()

# Display the generated image
plt.figure(figsize=(8, 8))
plt.imshow(binary_image, cmap='gray')
plt.axis('off')
plt.title('Binary Image of Root with Hairs')
plt.show()
