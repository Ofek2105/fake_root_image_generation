import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line_aa, bezier_curve
from skimage.draw import line as draw_line
from scipy.ndimage import binary_fill_holes
from skimage.draw import polygon
from typing import Tuple


class RootImageGenerator:
  def __init__(self, root_skele_points,
               root_width=5, root_width_std=1,
               hair_length=50, hair_length_std=50,
               hair_thickness=5, hair_thickness_std=1,
               hair_craziness=3,
               hair_n=100, img_width=1000, img_height=1000):
    self.points = root_skele_points

    self.root_width = root_width
    self.root_width_std = root_width_std

    self.hair_length = hair_length
    self.hair_length_std = hair_length_std

    self.hair_thickness = hair_thickness
    self.hair_thickness_std = hair_thickness_std

    self.hair_craziness = hair_craziness

    self.hair_n = hair_n
    self.img_width = img_width
    self.img_height = img_height
    self.normals, self.deltas = self.compute_normals()

  def compute_normals(self):
    normals = []
    deltas = []
    for i in range(len(self.points) - 1):
      dx = self.points[i + 1, 0] - self.points[i, 0]
      dy = self.points[i + 1, 1] - self.points[i, 1]
      norm = np.sqrt(dx ** 2 + dy ** 2)
      normals.append(np.array([-dy, dx]) / norm)
      deltas.append(np.array((dx / norm, dy / norm)))
    normals.append(normals[-1])
    deltas.append(deltas[-1])
    return np.array(normals), np.array(deltas)

  def generate_parallel_lines(self):
    parallel_lines = []
    for line_offset in [-self.root_width, self.root_width]:
      line = []
      for point, normal in zip(self.points, self.normals):
        offset = np.random.normal(line_offset, self.root_width_std)
        new_point = point + normal * offset
        line.append(new_point)
      parallel_lines.append(np.array(line))
    return parallel_lines

  def draw_lines_connections(self, bin_img, lin1, lin2):
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
      delta = self.deltas[i]

      if np.random.rand() > 0.5:
        normal = -normal

      hair_length = np.random.normal(self.hair_length, self.hair_length_std)
      if hair_length < 0:
        hair_length = -hair_length

      thickness = int(np.random.normal(self.hair_thickness, self.hair_thickness_std))

      cp_inc = normal * hair_length / 2 + np.random.normal(hair_length / 8, hair_length / 4, size=2)
      control_point = (point + cp_inc).astype(int)
      end_point = (point + normal * hair_length).astype(int)

      self.draw_thick_bezier_curve(bin_image.T, point, delta, control_point, end_point, thickness, self.hair_craziness)

  def draw_thick_bezier_curve(self,
                              bin_image: np.ndarray,
                              s1_point: Tuple[int, int],
                              s2_direction: Tuple[int, int],
                              m_point: Tuple[int, int],
                              e_point: Tuple[int, int],
                              thickness: int,
                              weight: int = 3):

    r0, c0 = s1_point  # Starting point of the first curve
    r1, c1 = m_point  # Control point (shared)
    r2, c2 = e_point

    r0b = int(r0 + thickness * s2_direction[0])
    c0b = int(c0 + thickness * s2_direction[1])

    rr, cc = bezier_curve(r0, c0, r1, c1, r2, c2, weight)
    rrb, ccb = bezier_curve(r0b, c0b, r1, c1, r2, c2, weight)
    poly_rr, poly_cc = polygon(np.concatenate([rr, rrb[::-1]]), np.concatenate([cc, ccb[::-1]]))
    valid = (poly_rr >= 0) & (poly_rr < self.img_height) & (poly_cc >= 0) & (poly_cc < self.img_width)
    bin_image[poly_rr[valid], poly_cc[valid]] = True

  def generate(self):
    line1, line2 = self.generate_parallel_lines()
    root_image_bi = np.zeros((self.img_height, self.img_width), dtype=np.uint8)

    self.draw_lines_connections(root_image_bi, line1, line2)
    root_image_bi = binary_fill_holes(root_image_bi)

    hairs_image_bi = np.zeros((self.img_height, self.img_width), dtype=np.uint8)
    self.draw_hairs(hairs_image_bi)  # Draw hair-like lines randomly along the root

    merged_image = np.logical_or(root_image_bi, hairs_image_bi).astype(np.uint8)

    return merged_image, root_image_bi, hairs_image_bi


np.random.seed(0)
# Usage example
x = np.linspace(50, 200, 500)  # More main_root_points for a smoother curve
y = 100 + np.sin(x / 50) * 20 + np.sin(x / 100) * 40  # Combination of sine waves
y = 0 + 4 * x

points = np.vstack((x, y)).T

# Initialize the RootImageGenerator with the main_root_points and other desired parameters
root_image_generator = RootImageGenerator(points)

# Generate the binary image using the generate method of the RootImageGenerator class
binary_image, root_only, hairs_only = root_image_generator.generate()

# Display the generated image
plt.figure(figsize=(8, 8))
plt.imshow(binary_image, cmap='gray')
plt.axis('off')
plt.title('Binary Image of Filled Shape')
plt.show()

