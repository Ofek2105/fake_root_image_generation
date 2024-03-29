import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line_aa, bezier_curve
from skimage.draw import line as draw_line
from scipy.ndimage import binary_fill_holes


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
    # Draw lines along each side of the root
    for lin in [lin1, lin2]:
        for i in range(len(lin) - 1):
            start = np.clip(lin[i], 0, [self.img_width - 1, self.img_height - 1]).astype(int)
            end = np.clip(lin[i + 1], 0, [self.img_width - 1, self.img_height - 1]).astype(int)
            rr, cc = draw_line(start[1], start[0], end[1], end[0])
            bin_img[rr, cc] = 1

    # Close the root shape by connecting the ends of lin1 and lin2
    start_top = np.clip(lin1[0], 0, [self.img_width - 1, self.img_height - 1]).astype(int)
    end_top = np.clip(lin2[0], 0, [self.img_width - 1, self.img_height - 1]).astype(int)
    rr, cc = draw_line(start_top[1], start_top[0], end_top[1], end_top[0])
    bin_img[rr, cc] = 1

    start_bottom = np.clip(lin1[-1], 0, [self.img_width - 1, self.img_height - 1]).astype(int)
    end_bottom = np.clip(lin2[-1], 0, [self.img_width - 1, self.img_height - 1]).astype(int)
    rr, cc = draw_line(start_bottom[1], start_bottom[0], end_bottom[1], end_bottom[0])
    bin_img[rr, cc] = 1


  def draw_hairs(self, bin_image):
    for i in range(0, len(self.points), max(1, len(self.points) // self.hair_n)):
        point = np.clip(self.points[i], 0, [self.img_width - 1, self.img_height - 1]).astype(int)
        normal = self.normals[i]

        if np.random.rand() > 0.5:
            normal = -normal

        length_variation = np.random.normal(0, self.hair_length * 0.2)
        length = max(5, self.hair_length + length_variation)  # Ensure hair length is not too short
        thickness_variation = np.random.normal(0, self.hair_thickness * 0.2)
        thickness = max(1, self.hair_thickness + thickness_variation)

        # Create a curvature for the hair
        control_point = (point + normal * length / 2 + np.random.normal(0, length / 4, size=2)).astype(int)
        end_point = (point + normal * length).astype(int)

        # Draw bezier curve for hair
        rr, cc = bezier_curve(point[1], point[0], control_point[1], control_point[0], end_point[1], end_point[0], thickness)
        valid = (rr >= 0) & (rr < self.img_height) & (cc >= 0) & (cc < self.img_width)
        bin_image[rr[valid], cc[valid]] = 1


  def generate(self):
    line1, line2 = self.generate_parallel_lines()
    binary_image = np.zeros((self.img_height, self.img_width), dtype=np.uint8)

    self.draw_connections(binary_image, line1, line2)
    binary_image = binary_fill_holes(binary_image)  # Fill holes to make it look like a single root
    self.draw_hairs(binary_image)  # Draw hair-like lines randomly along the root
    return binary_image


# Usage example
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
