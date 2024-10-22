import math
import random

from image_processing_methods.IP_funcs import (
    apply_alpha_blending,
    add_light_effect)

import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line as draw_line
from SynDataGeneration.my_bezier_curve import draw_my_thick_bezier_curve_old
from SynDataGeneration.my_bezier_curve import draw_root_end_bezier_curve, draw_hair_random_walk
from image_processing_methods.IP_funcs import get_polygons_bbox_from_bin_image
from SynDataGeneration.gen_main_root_points import generator_main_roots
import cv2
from soilGeneration.soil_generation import gen_soil_image


def plot_normals_and_deltas_with_matplotlib(image, points, normals, deltas, scale=10):
    """
    Plots the normals and deltas on the image using matplotlib.

    Parameters:
    - image: The image on which to draw the arrows.
    - points: The points on the line.
    - normals: The normals at each point.
    - deltas: The deltas at each point.
    - scale: Scaling factor for the arrow length.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap='gray')
    head_width = 1
    head_length = 1
    for point, normal, delta in zip(points, normals, deltas):
        start_point = point.astype(int)
        normal_end_point = point + normal * scale
        delta_end_point = point + delta * scale

        ax.plot(start_point[0], start_point[1], 'ro', markersize=scale)

        # Draw normal arrow (green)
        ax.arrow(start_point[0], start_point[1],
                 normal_end_point[0] - start_point[0],
                 normal_end_point[1] - start_point[1],
                 head_width=head_width, head_length=head_length, fc='green', ec='green')

        # Draw delta arrow (blue)
        ax.arrow(start_point[0], start_point[1],
                 delta_end_point[0] - start_point[0],
                 delta_end_point[1] - start_point[1],
                 head_width=head_width, head_length=head_length, fc='blue', ec='blue')

    ax.set_title('Image with Normals (green) and Deltas (blue)')
    plt.axis('on')
    plt.show()


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
    def __init__(self, root_skele_points, hair_type="random_walk",
                 root_width=5, root_width_std=1,
                 hair_length=20, hair_length_std=20,
                 hair_thickness=5, hair_thickness_std=1,
                 hair_craziness=0,
                 root_start_percent=0.1, root_end_percent=0.1,
                 hair_density=0.09, img_width=300, img_height=300):

        self.points = root_skele_points
        if len(self.points) < 2:
            raise Exception("Cannot Generate Image with less than 2 points")

        self.no_hair_area_start = math.ceil(root_start_percent * len(self.points))
        self.no_hair_area_end = math.ceil(root_end_percent * len(self.points))

        self.hair_type = hair_type

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
        self.root_mask = np.zeros((self.img_height, self.img_width), dtype=np.uint8)
        self.hairs_mask = np.zeros((self.img_height, self.img_width), dtype=np.uint8)

    def compute_normals(self, window_size=2):
        if window_size < 2:
            raise ValueError("window_size must be at least 2")

        normals = []
        deltas = []

        for i in range(len(self.points) - window_size):
            current_point = self.points[i]
            if window_size == 2:
                segment = self.points[i + window_size]
            else:
                segment = np.mean(self.points[i + 1:i + window_size], axis=1)

            dx = current_point[0] - segment[0]
            dy = current_point[1] - segment[1]
            norm = np.sqrt(dx ** 2 + dy ** 2)
            delta = np.array((dx / norm, dy / norm))

            # Calculate normals
            norm_vec = np.array([-dy, dx]) / norm
            normals.append(norm_vec)
            deltas.append(delta)

        # Extend to match the original length
        normals.extend([normals[-1]] * (len(self.points) - len(normals)))
        deltas.extend([deltas[-1]] * (len(self.points) - len(deltas)))

        return np.array(normals), np.array(deltas)

    def generate_parallel_lines(self):

        root_start_pos = self.no_hair_area_start
        root_end_pos = len(self.points) - self.no_hair_area_end

        line1 = []
        line2 = []
        half_width = self.root_width / 2
        half_width_std = self.root_width_std / 2
        self.offset1_list = []
        for idx in range(len(self.points)):
            point = self.points[idx]
            normal = self.normals[idx]

            if idx < root_start_pos:
                offset1 = np.random.normal(idx * half_width / (root_start_pos + 1), half_width_std)
                offset2 = np.random.normal(idx * half_width / (root_start_pos + 1), half_width_std)
            elif idx > root_end_pos:
                offset1 = np.random.normal((len(self.points) - idx) * half_width / self.no_hair_area_end,
                                           half_width_std)
                offset2 = np.random.normal((len(self.points) - idx) * half_width / self.no_hair_area_end,
                                           half_width_std / 2)
            else:
                offset1 = np.random.normal(half_width, half_width_std)
                offset2 = np.random.normal(half_width, half_width_std)

            self.offset1_list.append(offset1)
            new_point1 = point + normal * offset1
            new_point2 = point - normal * offset2
            line1.append(new_point1)
            line2.append(new_point2)

        return [line1, line2]

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

    def draw_hairs(self):
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

            hair_length = np.random.normal(self.hair_length + self.root_width + self.root_width_std,
                                           self.hair_length_std)

            if hair_length < self.root_width + self.root_width_std:
                continue

            thickness = int(np.random.normal(self.hair_thickness, self.hair_thickness_std))

            cp_inc = normal * hair_length / 2 + np.random.normal(hair_length / 8, hair_length / 4, size=2)
            control_point = (point + cp_inc).astype(int)
            end_point = (point + normal * hair_length).astype(int)

            if self.hair_type == "bezier":
                polygon_, bbox_ = draw_my_thick_bezier_curve_old(self.hairs_mask, point,
                                                                 control_point, delta,
                                                                 end_point, thickness, self.hair_craziness,
                                                                 neg_mask=self.root_mask)
            elif self.hair_type == "random_walk":
                polygon_, bbox_ = draw_hair_random_walk(self.hairs_mask, point, normal, hair_length,
                                                        momentum=self.hair_craziness,
                                                        initial_width=thickness, step_size=3, neg_mask=self.root_mask)
            else:
                raise NotImplementedError("hair type not defined, choose 'bezier' or 'random_walk'")

            if len(bbox_) == 0:
                continue

            hairs_bboxes.append(bbox_)  # TODO: remove bbox... not needed
            hairs_polygons.append(polygon_)
            actual_root_count += 1

        return actual_root_count, hairs_polygons, hairs_bboxes

    def draw_main_root(self, line1, line2):
        # Convert line points to integer coordinates
        line1 = np.array(line1, dtype=np.int32)
        line2 = np.array(line2, dtype=np.int32)

        # Create a combined array of the points forming the closed shape
        combined_shape = np.concatenate([line1, line2[::-1]])

        # Draw and fill the polygon on the binary image
        cv2.fillPoly(self.root_mask, [combined_shape], color=1)

    def add_root_darker_middle_effect(self, root_hair_image, apply_chane=0.5):
        if random.random() > apply_chane:
            return root_hair_image

        white_rgb = np.array((255, 255, 255))
        color = white_rgb * random.uniform(0.8, 0.99)

        width_list = np.clip(self.offset1_list, 0.5, None)
        for i, point in enumerate(self.points):
            radius = np.random.uniform(width_list[i] * 0.43, width_list[i] * 0.6)
            deform_x = random.uniform(0.8, 1.2)
            deform_y = random.uniform(0.8, 1.2)

            x, y = point
            axes = (int(radius * deform_x), int(radius * deform_y))



            cv2.ellipse(root_hair_image, (x, y), axes, 0, 0, 360, color, -1)

        return root_hair_image


    def generate(self, new_shape=None, add_soil=True, add_flare=True, add_blurr=True):

        line1, line2 = self.generate_parallel_lines()
        self.draw_main_root(line1, line2)

        root_poly, root_bbox = get_polygons_bbox_from_bin_image(self.root_mask)

        hair_num, hairs_poly, hairs_bbox = self.draw_hairs()

        merged_mask = np.logical_or(self.root_mask, self.hairs_mask).astype(np.uint8)
        gray_intensity_factor = np.random.rand() * 0.5 + 0.5  # random number between 0.5 and 1
        color_image = cv2.cvtColor((merged_mask * 255 * gray_intensity_factor).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        root_hair_image = self.add_root_darker_middle_effect(color_image, apply_chane=0.7)

        if add_soil:
            soil_image = gen_soil_image((self.img_width, self.img_height),
                                        soil_images_folder_path='soilGeneration/background_resources')

            merged_mask = apply_alpha_blending(root_hair_image, soil_image)

            if add_flare and random.random() < 1:
                merged_mask = add_light_effect(merged_mask)
            if add_blurr:
                blur_strength = np.random.choice([5, 7, 11, 15, 31])
                merged_mask = cv2.GaussianBlur(merged_mask, (blur_strength, blur_strength), 0)

        if new_shape is not None:
            merged_mask = cv2.resize(merged_mask, new_shape, interpolation=cv2.INTER_AREA)
            # self.root_mask = resize(merged_mask, new_shape)
            self.hairs_mask = cv2.resize(self.hairs_mask, new_shape, interpolation=cv2.INTER_AREA)

        output = {  # TODO: remove useless outputs
            "full image": merged_mask,
            "only roots": merged_mask,
            "only hairs": self.hairs_mask,
            "hair count": hair_num,
            "hairs polygons": hairs_poly,
            "hairs bbox": hairs_bbox,
            "Main root bbox": root_bbox,
            "Main root polygon": root_poly
        }

        return output


if __name__ == '__main__':
    np.random.seed(0)

    rect_out = (50, 50, 550, 550)
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
        cv2.imshow("image", properties["full image"] * 255)
        cv2.waitKey(0)

        image = np.zeros((params['img_height'], params['img_width'], 3), dtype=np.uint8)
        plot_normals_and_deltas_with_matplotlib(image, root_image_class.points,
                                                root_image_class.normals, root_image_class.deltas, scale=2)
