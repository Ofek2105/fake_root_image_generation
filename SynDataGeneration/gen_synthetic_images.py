import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line as draw_line
from skimage.transform import resize
from scipy.ndimage import binary_fill_holes
from SynDataGeneration.my_bezier_curve import draw_my_thick_bezier_curve, draw_single_root_end
from image_processing_methods.IP_funcs import get_polygons_bbox_from_bin_image
from SynDataGeneration.gen_main_root_points import generator_main_roots
import cv2


def plot_bin_root_hairs(properties):
  fig, axes = plt.subplots(1, 3, figsize=(15, 5))
  fig.suptitle(f'{properties["hair count"]} hairs')
  axes[0].imshow(properties["full image"], cmap='gray')
  axes[0].axis('off')
  axes[0].set_title('Bin image')
  axes[1].imshow(properties["only roots"], cmap='gray')
  axes[1].axis('off')
  axes[1].set_title('root only')
  axes[2].imshow(properties["only hairs"], cmap='gray')
  axes[2].axis('off')
  axes[2].set_title('hair only')
  plt.show()


class RootImageGenerator:
  def __init__(self, root_skele_points,
               root_width=5, root_width_std=1,
               hair_length=20, hair_length_std=20,
               hair_thickness=5, hair_thickness_std=1,
               hair_craziness=0,
               hair_density=0.09, img_width=300, img_height=300):

    self.points = root_skele_points
    if len(self.points) < 2:
      raise Exception("Cannot Generate Image with less than 2 points")

    self.no_hair_area_start = 5
    self.no_hair_area_end = 5

    self.root_width = root_width
    self.root_width_std = root_width_std

    self.hair_length = hair_length
    self.hair_length_std = hair_length_std

    self.hair_thickness = hair_thickness
    self.hair_thickness_std = hair_thickness_std

    self.hair_craziness = hair_craziness

    self.hair_n = int((len(self.points) - self.no_hair_area_start - self.no_hair_area_end) * hair_density)
    self.img_width = img_width
    self.img_height = img_height

    self.normals, self.deltas = self.compute_normals()
    self.root_image_bi = np.zeros((self.img_height, self.img_width), dtype=np.uint8)

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

  def draw_lines_connections(self, lin1, lin2):

    for lin in [lin1, lin2]:
      for i in range(len(lin) - 1):
        start = np.clip(lin[i], 0, [self.img_width - 1, self.img_height - 1]).astype(int)
        end = np.clip(lin[i + 1], 0, [self.img_width - 1, self.img_height - 1]).astype(int)
        rr, cc = draw_line(start[1], start[0], end[1], end[0])
        self.root_image_bi[rr, cc] = 1

    # Close the root shape by connecting the ends of lin1 and lin2
    start_top = np.clip(lin1[0], 0, [self.img_width - 1, self.img_height - 1]).astype(int)
    end_top = np.clip(lin2[0], 0, [self.img_width - 1, self.img_height - 1]).astype(int)
    rr, cc = draw_line(start_top[1], start_top[0], end_top[1], end_top[0])
    self.root_image_bi[rr, cc] = 1

    start_bottom = np.clip(lin1[-1], 0, [self.img_width - 1, self.img_height - 1]).astype(int)
    end_bottom = np.clip(lin2[-1], 0, [self.img_width - 1, self.img_height - 1]).astype(int)
    rr, cc = draw_line(start_bottom[1], start_bottom[0], end_bottom[1], end_bottom[0])
    self.root_image_bi[rr, cc] = 1

  def gen_hair_pos(self):

    start = self.no_hair_area_start
    end = len(self.points) - self.no_hair_area_end

    if self.hair_n <= 0 or start >= end:
      return

    if self.hair_n > 1:
      step = (end - start) // (self.hair_n - 1)
    else:
      step = end - start

    for i in range(start, end, max(1, step)):
      yield i

  def draw_hairs(self, bin_image):
    actual_root_count = 0
    hairs_bboxes = []
    hairs_polygons = []

    if self.hair_n == 0:
      return 0, [], []

    for i in self.gen_hair_pos():
      point = np.clip(self.points[i],
                      0, [self.img_width - 1, self.img_height - 1]).astype(int)
      normal = self.normals[i]
      delta = self.deltas[i]

      if np.random.rand() > 0.5:
        normal = -normal

      hair_length = np.random.normal(self.hair_length, self.hair_length_std)

      if hair_length < self.root_width + self.root_width_std:
        continue

      if hair_length < self.root_width + 5 * self.root_width_std:  # to fix unusually large std
        continue

      thickness = int(np.random.normal(self.hair_thickness, self.hair_thickness_std))

      cp_inc = normal * hair_length / 2 + np.random.normal(hair_length / 8, hair_length / 4, size=2)
      control_point = (point + cp_inc).astype(int)
      end_point = (point + normal * hair_length).astype(int)

      polygon_, bbox_ = draw_my_thick_bezier_curve(bin_image, point,
                                                   control_point,
                                                   end_point, thickness, self.hair_craziness,
                                                   neg_mask=self.root_image_bi)
      if len(bbox_) == 0:
        # print("could not find the polygon")
        continue

      hairs_bboxes.append(bbox_)
      hairs_polygons.append(polygon_)
      actual_root_count += 1

    return actual_root_count, hairs_polygons, hairs_bboxes

  def draw_root_ends(self, line1, line2):
    start1, start2 = line1[0], line2[0]
    end1, end2 = line1[-1], line2[-1]
    start_root_dir = -1 * self.deltas[0]
    end_root_dir = self.deltas[-1]

    draw_single_root_end(self.root_image_bi, start1, start2, start_root_dir, thickness_=self.root_width)
    draw_single_root_end(self.root_image_bi, end1, end2, end_root_dir, thickness_=self.root_width)

  def generate(self, new_shape=None):
    line1, line2 = self.generate_parallel_lines()

    self.draw_lines_connections(line1, line2)
    self.root_image_bi = binary_fill_holes(self.root_image_bi)
    self.draw_root_ends(line1, line2)

    root_poly, root_bbox = get_polygons_bbox_from_bin_image(self.root_image_bi)

    hairs_image_bi = np.zeros((self.img_height, self.img_width), dtype=np.uint8)

    hair_num, hairs_poly, hairs_bbox = self.draw_hairs(hairs_image_bi)

    merged_image = np.logical_or(self.root_image_bi, hairs_image_bi).astype(np.uint8)

    if new_shape is not None:
      merged_image = resize(merged_image, new_shape)
      self.root_image_bi = resize(merged_image, new_shape)
      hairs_image_bi = resize(hairs_image_bi, new_shape)

    output = {
      "full image": merged_image,
      "only roots": self.root_image_bi,
      "only hairs": hairs_image_bi,
      "hair count": hair_num,
      "hairs polygons": hairs_poly,
      "hairs bbox": hairs_bbox,
      "Main root bbox": root_bbox,
      "Main root polygon": root_poly
    }
    return output


if __name__ == '__main__':
  np.random.seed(0)

  rect_out = (50, 50, 250, 250)
  delt = 20
  rect_in = (rect_out[0] + delt, rect_out[1] + delt, rect_out[2] - delt, rect_out[3] - delt)

  params = {
    "root_width": 10,
    "root_width_std": 1,
    "hair_length": 30,
    "hair_length_std": 10,
    "hair_thickness": 1,
    "hair_thickness_std": 0,
    "hair_craziness": 0,
    "hair_density": 0.3,
    "img_width": 300,
    "img_height": 300
  }

  for i, main_root_points in enumerate(generator_main_roots(5)):
    root_image_class = RootImageGenerator(main_root_points, **params)
    properties = root_image_class.generate()
    print("image")
    # save image
    image_name = f'im{i}_haircount_{properties["hair count"]}.png'
    cv2.imwrite(f'temp_folder/{image_name}', properties["full image"] * 255)

    # Display the generated image
    # plt.figure(figsize=(8, 8))
    # plt.imshow(properties["full image"], cmap='gray')
    # plt.axis('off')
    # plt.title(f'Binary Image of Filled Shape\nHair Count {properties["hair count"]}')
    # plt.show()
    # break
