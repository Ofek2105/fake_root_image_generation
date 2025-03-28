import os
import numpy as np
from PIL import Image

from SynDataGeneration.gen_main_root_points import generator_main_roots
from SynDataGeneration.gen_synthetic_images import RootImageGenerator
from image_processing_methods.IP_funcs import fig_to_array, save_pipline_image
from coco_json_initialization import COCODatasetGenerator
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import itertools
from tqdm import tqdm
from shapely.geometry import Polygon
import pathlib
import time


# def plot_fake_and_pred_images(fake_im_properties, contours, file_path=None):
#     full_image = fake_im_properties["full image"]
#     hairs_polygons = fake_im_properties["hairs polygons"]
#     hairs_bbox = fake_im_properties["hairs bbox"]
#     root_bbox = fake_im_properties["Main root bbox"]
#     root_polygon = fake_im_properties["Main root polygon"]
#     font_size_ = 20
#     linewidth_poly = 8
#     linewidth_bbox = 5
#     n_plots = 3 if contours is not None else 2
#
#     # Create a figure and subplots (now three subplots)
#     fig, ax = plt.subplots(1, n_plots, figsize=(90, 30))
#     ax[0].imshow(full_image, cmap='gray')
#     ax[0].set_title("Synthetic root", size=font_size_)
#
#     ax[1].imshow(full_image, cmap='gray')
#     ax[1].set_title("Annotated Synthetic root", size=font_size_)
#
#     root_poly_points = scale_up_polygon(root_polygon, full_image.shape)
#     ax[1].plot(root_poly_points[:, 0], root_poly_points[:, 1], 'g-',
#                linewidth=linewidth_poly, label='Main Root polygon')
#
#     root_bbox = scale_up_bbox(root_bbox, full_image.shape)
#     rect = patches.Rectangle((root_bbox[0], root_bbox[1]), root_bbox[2], root_bbox[3],
#                              linewidth=linewidth_bbox, edgecolor='purple',
#                              facecolor='none', label='Main Root BBox')
#     ax[1].add_patch(rect)
#
#     # Plot bounding boxes for hairs
#     for bbox in hairs_bbox:
#         hair_bbox = scale_up_bbox(bbox, full_image.shape)
#         rect = patches.Rectangle((hair_bbox[0], hair_bbox[1]), hair_bbox[2], hair_bbox[3],
#                                  linewidth=linewidth_bbox, edgecolor='r', facecolor='none')
#         ax[1].add_patch(rect)
#
#     # Plot polygons for hairs (each polygon is a sequence of x, y coordinates)
#     for polygon in hairs_polygons:
#         poly_points = scale_up_polygon(polygon, full_image.shape)
#         ax[1].plot(poly_points[:, 0], poly_points[:, 1], color='blue',
#                    linewidth=3)
#
#     # Draw contours on the third axis
#     if contours is not None:
#         ax[2].imshow(full_image, cmap='gray')
#         ax[2].set_title("Found hair tips", size=font_size_)
#         ax[2].imshow(full_image, cmap='gray')
#         for contour in contours:
#             contour = contour.reshape(-1, 2)  # Reshape to (n, 2)
#             ax[2].plot(contour[:, 0], contour[:, 1], color='yellow', linewidth=3)
#
#         ax[0].set_axis_off()
#         ax[1].set_axis_off()
#         ax[2].set_axis_off()
#
#     if file_path is not None:
#         plt.savefig(file_path, bbox_inches='tight')
#     else:
#         plt.show()


def save_binary_image(binary_image, base_path, params, image_id, hair_count=None):
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # Constructing the filename
    # file_name = (
    #     f"rw{params['root_width']}_rs{params['root_width_std']}_hl{params['hair_length']}"
    #     f"_ls{params['hair_length_std']}_ht{params['hair_thickness']}_ts{params['hair_thickness_std']}"
    #     f"_hc{params['hair_craziness']}_hd{int(params['hair_density'] * 100)}_w{params['img_width']}"
    #     f"_h{params['img_height']}_{image_id}.png")

    file_name = f"{str(time.time()).replace('.', '')}_{image_id}.png"
    file_path = os.path.join(base_path, file_name)
    if len(binary_image.shape) == 3:  # not bin_image
        image = Image.fromarray(binary_image)
    else:
        image = Image.fromarray((binary_image * 255).astype(np.uint8))
    image.save(file_path)

    return file_path, file_name


def calculate_polygon_area(coords, image_shape):
    if len(coords) < 6:  # Not a polygon
        return 0

    # Convert the flat list of coordinates to a list of (x, y) tuples
    points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
    polygon = Polygon(points)
    normalized_area = polygon.area
    return normalized_area * image_shape[0] * image_shape[1]


def scale_up_bbox(bbox, original_shape):
    x, y, w, h = bbox
    x = int(np.round(x * original_shape[0]))
    y = int(np.round(y * original_shape[1]))
    w = int(np.round(w * original_shape[0]))
    h = int(np.round(h * original_shape[1]))
    return [x, y, w, h]


def scale_up_polygon(polygon, original_shape):
    root_poly_points = np.array(polygon).reshape(-1, 2)
    root_poly_points[:, 0] = np.round(root_poly_points[:, 0] * original_shape[0])
    root_poly_points[:, 1] = np.round(root_poly_points[:, 1] * original_shape[1])
    return root_poly_points.astype(int)


def plot_with_annotations(properties, save_image_path=None, plot_image=True):
    # Retrieve the full image and other necessary details from fake_im_properties
    full_image = properties["full image"]
    hairs_polygons = properties["hairs polygons"]
    hairs_bbox = properties["hairs bbox"]
    root_bbox = properties["Main root bbox"]
    root_polygon = np.array(properties["Main root polygon"]).reshape((-1, 2))
    font_size_ = 11
    linewidth_poly = 1
    linewidth_bbox = 1

    # Create a figure and a subplot
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(full_image, cmap='gray')
    ax[1].imshow(full_image, cmap='gray')
    ax[0].set_title("Synthetic root", size=font_size_)
    ax[1].set_title("Annotated Synthetic root", size=font_size_)

    # root_poly_points = scale_up_polygon(root_polygon, full_image.shape)

    ax[1].plot(root_polygon[:, 0], root_polygon[:, 1], 'g-',
               linewidth=linewidth_poly, label='Main Root polygon')

    # root_bbox = scale_up_bbox(root_bbox, full_image.shape)
    rect = patches.Rectangle((root_bbox[0], root_bbox[1]), root_bbox[2], root_bbox[3],
                             linewidth=linewidth_bbox, edgecolor='purple',
                             facecolor='none', label='Main Root BBox')
    ax[1].add_patch(rect)

    # Plot bounding boxes for hairs
    for bbox in hairs_bbox:
        # hair_bbox = scale_up_bbox(bbox, full_image.shape)
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                 linewidth=linewidth_bbox, edgecolor='r', facecolor='none')
        ax[1].add_patch(rect)

    # Plot polygons for hairs (each polygon is a sequence of x, y coordinates)
    for polygon in hairs_polygons:
        # poly_points = scale_up_polygon(polygon, full_image.shape)
        polygon_s = np.array(polygon).reshape((-1, 2))
        ax[1].plot(polygon_s[:, 0], polygon_s[:, 1], color='blue',
                   linewidth=linewidth_poly)

    ax[0].set_axis_off()
    ax[1].set_axis_off()
    if save_image_path is not None:
        save_pipline_image(fig_to_array(fig, 500), save_image_path)
    if plot_image:
        plt.show()
    plt.close()


def save_annotation_yolo(prop_: dict, base_path: str, file_name: str):
    p = pathlib.Path(base_path)
    p.mkdir(parents=True, exist_ok=True)

    with open(f'{base_path}\\{file_name}', 'w') as f:
        row = f'1 {" ".join([str(val) for val in prop_["Main root polygon"]])}\n'
        f.write(row)

        for i in range(prop_["hair count"]):
            row = f'0 {" ".join([str(val) for val in prop_["hairs polygons"][i]])}\n'
            f.write(row)


def annotate_specific_image_coco(coco, properties):
    timestamp = str(time.time()).replace('.', '')
    file_name = f"{timestamp}_{coco.image_id}.jpeg"
    file_path = os.path.join(coco.output_dir, file_name)
    height, width, _ = properties["full image"].shape
    im_id = coco.add_image(file_name, width=width, height=height)

    image = Image.fromarray(properties["full image"])
    image.save(file_path, quality=80)

    if properties["shifted_images"] is not None:
        for i, image in enumerate(properties["shifted_images"]):
            shifted_image_file_name = f"{timestamp}_{coco.image_id}_{i}.jpeg"
            shifted_image_file_path = os.path.join("shifted_images_datasets\\", shifted_image_file_name)
            PIL_image = Image.fromarray(image)

            PIL_image.save(shifted_image_file_path, quality=80)

    if len(properties["Main root polygon"]) != 0:
        coco.add_annotation(image_id=im_id, category_id=1, segmentation=properties["Main root polygon"])  # root

    for polygon in properties["hairs polygons"]:
        coco.add_annotation(image_id=im_id, category_id=0, segmentation=polygon)  # hair


def annotate_specific_params_coco(coco, images_per_root, n_unique_roots, params, save_shifted_images=False):
    rect_out_ = (
        params["img_width"] * 0.1, params["img_height"] * 0.1, params["img_width"] * 0.9, params["img_height"] * 0.9)

    for main_root_points in generator_main_roots(n_unique_roots, rect_out_):

        for _ in range(images_per_root):
            root_image_class = RootImageGenerator(main_root_points, **params)
            properties = root_image_class.generate(add_shifted_images=save_shifted_images)
            annotate_specific_image_coco(coco, properties)


def annotate_specific_params_yolo(images_per_root, n_unique_roots, params):
    rect_out_ = (
        params["img_width"] * 0.1, params["img_height"] * 0.1, params["img_width"] * 0.9, params["img_height"] * 0.9)
    for main_root_points in generator_main_roots(n_unique_roots, rect_out_):

        for _ in range(images_per_root):
            root_image_class = RootImageGenerator(main_root_points, **params)
            properties = root_image_class.generate()
            _, filename = save_binary_image(properties["full image"], "dataset\\images", params,
                                            annotate_specific_params_yolo.counter_id,
                                            hair_count=properties["hair count"])
            filename_txt = filename.split('.')[0] + ".txt"
            save_annotation_yolo(properties, "dataset\\labels", filename_txt)
            annotate_specific_params_yolo.counter_id += 1


def create_dataset(all_params, n_main_root_=3, hair_gen_per_main_root_=3, dataformat="coco", save_shifted_images=False):
    cocoGen = COCODatasetGenerator('dataset\\')

    iteration_num = count_iterations(all_params)
    annotate_specific_params_yolo.counter_id = 0

    for params in tqdm(generator_combination_dict(all_params), total=iteration_num):
        if dataformat == "yolo":
            raise ("BRO! I changed the code so the polygons and bboxes are absolute values (not between 0 and 1)\n "
                   "if you want yolo format again you need to change it here so it would work with everything")
            # annotate_specific_params_yolo(images_per_root=hair_gen_per_main_root_,
            #                               n_unique_roots=n_main_root_, params=params)
        if dataformat == "coco":
            annotate_specific_params_coco(cocoGen, images_per_root=hair_gen_per_main_root_,
                                          n_unique_roots=n_main_root_, params=params,
                                          save_shifted_images=save_shifted_images)
    cocoGen.save_annotations()


def generator_combination_dict(possibilities_dict):
    keys = []
    values = []

    for key, value in possibilities_dict.items():
        if isinstance(value, list):
            keys.append(key)
            values.append(value)

    for combination in itertools.product(*values):
        params = {}
        for key, value in possibilities_dict.items():
            if key in keys:
                index = keys.index(key)
                params[key] = combination[index]
            else:
                params[key] = value
        yield params


def count_iterations(possibilities_dict):
    iterations = 1
    for key, value in possibilities_dict.items():
        if isinstance(value, list):
            iterations *= len(value)
    return iterations


def show_images():
    # np.random.seed(21)
    print("Generating and Plotting images")
    N = 10

    params = {
        "root_width": 40,
        "root_width_std": 3,
        "hair_length": 70,
        "hair_length_std": 30,
        "hair_thickness": 5,
        "hair_thickness_std": 2,
        "hair_craziness": 0.97,  # 0 or 1
        "hair_density": 0.3,
        "img_width": 960,
        "img_height": 960,
        "root_start_percent": 0.20,
        "root_end_percent": 0.05,
        "hair_type": "random_walk",  # ["bezier", "random_walk-walk"]
        "background_type": "real"  # ["real", "perlin"]'
    }

    # rect_out_ = (50, 50, 250, 250)
    rect_out_ = (
        params["img_width"] * 0.1, params["img_height"] * 0.1, params["img_width"] * 0.9, params["img_height"] * 0.9)

    for main_root_points in generator_main_roots(N, rect=rect_out_):
        root_image_class = RootImageGenerator(main_root_points, **params)
        properties = root_image_class.generate(save_pipline_path="pipline_images_save_dump")
        plot_with_annotations(properties, plot_image=False, save_image_path="pipline_images_save_dump")


if __name__ == '__main__':
    # possibilities = {
    #     "root_width": [20, 10, 40],
    #     "root_width_std": [1, 3],
    #     "hair_length": [3, 20, 50],
    #     "hair_length_std": [30],
    #     "hair_thickness": [3, 5],
    #     "hair_thickness_std": [2, 4],
    #     "hair_craziness": [0.85, 0.97],  # 0 or 1
    #     "hair_density": [0.3, 0.1],
    #     "img_width": 960,
    #     "img_height": 960,
    #     "root_start_percent": [0.05],
    #     "root_end_percent": [0.15],
    #     "hair_type": "random_walk",  # ["bezier", "random_walk-walk"]'
    #     "background_type": ["real", "perlin"]   # ["real", "perlin"]'
    # }
    #  changed the width and height from 960 to 1920 for the subpixel shift

    possibilities = {
        "root_width": [40, 20, 80],
        "root_width_std": [2, 6],
        "hair_length": [6, 40, 100],
        "hair_length_std": [60],
        "hair_thickness": [6, 10],
        "hair_thickness_std": [4, 8],
        "hair_craziness": [0.85, 0.97],  # 0 or 1
        "hair_density": [0.3, 0.1],
        "img_width": 1920,
        "img_height": 1920,
        "root_start_percent": [0.05],
        "root_end_percent": [0.15],
        "hair_type": "random_walk",  # ["bezier", "random_walk-walk"]'
        "background_type": ["real", "perlin"]  # ["real", "perlin"]'
    }

    n_main_root = 10
    hair_gen_per_main_root = 3
    print(f'Number of Images to generate: {count_iterations(possibilities) * n_main_root * hair_gen_per_main_root}')
    # show_images()
    create_dataset(possibilities, n_main_root, hair_gen_per_main_root, save_shifted_images=True)

    # 12/10/24
    # It seems that the model performance is worse for roots that are grayer or slightly transparent.
    # also I fill like all the root segmentation has the same width, look into it.
    # TODO: add some random gray-scale for root drawing ...CHECK
    # TODO: add some random transparency for root drawing ...CHECK
    # TODO: add some random slight swirl in an image ... CHECK
    # TODO: add channel jitter ...CHECK

    #   26/10/24
    # Removed swirl effect so it won't hurt the annotations
    # we will save images in higher resolution 960 by 960

    # 10/12/2024
    # * fixed problem where all the hairs didn't got segmented
    # * added coco support
    # * images saved as jpeg with quality 90 (used PIL for saving)
    # * fixed root segmentation problem when root is looping into it self
    # * root and hairs segmentations requre less points (hairs rdp epsilon of 1; root rdp epsilon of 3)
    # * added new background method
    #