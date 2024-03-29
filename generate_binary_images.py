import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line as draw_line
from scipy.ndimage import binary_fill_holes
import math


class RootImageGenerator:
  def __init__(self, points, root_width=10, epsilon_std=1, hair_length=50, hair_thickness=2, hair_n=100, img_width=1000,
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
    normals = []
    for i in range(len(self.points) - 1):
      dx = self.points[i + 1, 0] - self.points[i, 0]
      dy = self.points[i + 1, 1] - self.points[i, 1]
      norm = np.sqrt(dx ** 2 + dy ** 2)
      normals.append(np.array([-dy, dx]) / norm)
    normals.append(normals[-1])
    return np.array(normals)

  def generate_parallel_lines(self):
    parallel_lines = []
    for line_offset in [-self.root_width, self.root_width]:
      line = []
      for point, normal in zip(self.points, self.normals):
        epsilon = np.random.normal(0, self.epsilon_std)
        offset = line_offset + epsilon
        new_point = point + normal * offset
        line.append(new_point)
      parallel_lines.append(np.array(line))
    return parallel_lines

  def draw_connections(self, bin_img, lin1, lin2):
    for lin in [lin1, lin2]:
        for i in range(len(lin) - 1):
            start = np.clip(lin[i], 0, [self.img_width - 1, self.img_height - 1]).astype(int)
            end = np.clip(lin[i + 1], 0, [self.img_width - 1, self.img_height - 1]).astype(int)
            rr, cc = draw_line(start[1], start[0], end[1], end[0])
            bin_img[rr, cc] = 1

    # Connect the ends of lin1 and lin2 to form a closed shape
    start = np.clip(lin1[0], 0, [self.img_width - 1, self.img_height - 1]).astype(int)
    end = np.clip(lin2[0], 0, [self.img_width - 1, self.img_height - 1]).astype(int)
    rr, cc = draw_line(start[1], start[0], end[1], end[0])
    bin_img[rr, cc] = 1

    start = np.clip(lin1[-1], 0, [self.img_width - 1, self.img_height - 1]).astype(int)
    end = np.clip(lin2[-1], 0, [self.img_width - 1, self.img_height - 1]).astype(int)
    rr, cc = draw_line(start[1], start[0], end[1], end[0])
    bin_img[rr, cc] = 1


  def draw_hairs(self, bin_image):
    for i in range(0, len(self.points), max(1, len(self.points) // self.hair_n)):
      point = np.clip(self.points[i], 0, [self.img_width - 1, self.img_height - 1]).astype(int)
      normal = self.normals[i]

      if np.random.rand() > 0.5:
        normal = -normal

      for j in range(-self.hair_thickness, self.hair_thickness + 1):
        length = np.random.normal(self.hair_length, self.hair_length * 0.5)
        angle = np.random.uniform(-math.pi / 18, math.pi / 18)
        rotation_matrix = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        rotated_normal = rotation_matrix.dot(normal)

        end_point = point + rotated_normal * length + rotated_normal[::-1] * j
        end_point = np.clip(end_point, 0, [self.img_width - 1, self.img_height - 1]).astype(int)

        rr, cc = draw_line(point[1], point[0], end_point[1], end_point[0])
        bin_image[rr, cc] = 1

  def generate(self):
    line1, line2 = self.generate_parallel_lines()
    binary_image = np.zeros((self.img_height, self.img_width), dtype=np.uint8)

    self.draw_connections(binary_image, line1, line2)
    binary_image = binary_fill_holes(binary_image)  # Fill holes to make it look like a single root
    self.draw_hairs(binary_image)  # Draw hair-like lines randomly along the root
    return binary_image


# Generate points for the root shape
x = np.linspace(50, 400, 500)  # More points for a smoother curve
y = 100 + np.sin(x / 50) * 20 + np.sin(x / 100) * 40  # Combination of sine waves

points = np.vstack((x, y)).T

# Initialize the RootImageGenerator with the points and other desired parameters
root_image_generator = RootImageGenerator(points)

# Generate the binary image using the generate method of the RootImageGenerator class
binary_image = root_image_generator.generate()

# Display the generated image
plt.figure(figsize=(8, 8))
plt.imshow(binary_image, cmap='gray')
plt.axis('off')
plt.title('Binary Image of Filled Shape')
plt.show()
