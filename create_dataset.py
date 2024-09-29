import os
import numpy as np
from PIL import Image

from SynDataGeneration.gen_main_root_points import generator_main_roots
from SynDataGeneration.gen_synthetic_images import RootImageGenerator, plot_bin_root_hairs
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from coco_json_initialization import CocoDataset
import itertools
from tqdm import tqdm
from shapely.geometry import Polygon
import pathlib
import time


def plot_fake_and_pred_images(fake_im_properties, contours, file_path=None):
    full_image = fake_im_properties["full image"]
    hairs_polygons = fake_im_properties["hairs polygons"]
    hairs_bbox = fake_im_properties["hairs bbox"]
    root_bbox = fake_im_properties["Main root bbox"]
    root_polygon = fake_im_properties["Main root polygon"]
    font_size_ = 20
    linewidth_poly = 8
    linewidth_bbox = 5
    n_plots = 3 if contours is not None else 2

    # Create a figure and subplots (now three subplots)
    fig, ax = plt.subplots(1, n_plots, figsize=(90, 30))
    ax[0].imshow(full_image, cmap='gray')
    ax[0].set_title("Synthetic root", size=font_size_)

    ax[1].imshow(full_image, cmap='gray')
    ax[1].set_title("Annotated Synthetic root", size=font_size_)

    root_poly_points = scale_up_polygon(root_polygon, full_image.shape)
    ax[1].plot(root_poly_points[:, 0], root_poly_points[:, 1], 'g-',
               linewidth=linewidth_poly, label='Main Root polygon')

    root_bbox = scale_up_bbox(root_bbox, full_image.shape)
    rect = patches.Rectangle((root_bbox[0], root_bbox[1]), root_bbox[2], root_bbox[3],
                             linewidth=linewidth_bbox, edgecolor='purple',
                             facecolor='none', label='Main Root BBox')
    ax[1].add_patch(rect)

    # Plot bounding boxes for hairs
    for bbox in hairs_bbox:
        hair_bbox = scale_up_bbox(bbox, full_image.shape)
        rect = patches.Rectangle((hair_bbox[0], hair_bbox[1]), hair_bbox[2], hair_bbox[3],
                                 linewidth=linewidth_bbox, edgecolor='r', facecolor='none')
        ax[1].add_patch(rect)

    # Plot polygons for hairs (each polygon is a sequence of x, y coordinates)
    for polygon in hairs_polygons:
        poly_points = scale_up_polygon(polygon, full_image.shape)
        ax[1].plot(poly_points[:, 0], poly_points[:, 1], color='blue',
                   linewidth=3)

    # Draw contours on the third axis
    if contours is not None:
        ax[2].imshow(full_image, cmap='gray')
        ax[2].set_title("Found hair tips", size=font_size_)
        ax[2].imshow(full_image, cmap='gray')
        for contour in contours:
            contour = contour.reshape(-1, 2)  # Reshape to (n, 2)
            ax[2].plot(contour[:, 0], contour[:, 1], color='yellow', linewidth=3)

        ax[0].set_axis_off()
        ax[1].set_axis_off()
        ax[2].set_axis_off()

    if file_path is not None:
        plt.savefig(file_path, bbox_inches='tight')
    else:
        plt.show()


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


def plot_with_annotations(properties):
    # Retrieve the full image and other necessary details from fake_im_properties
    full_image = properties["full image"]
    hairs_polygons = properties["hairs polygons"]
    hairs_bbox = properties["hairs bbox"]
    root_bbox = properties["Main root bbox"]
    root_polygon = properties["Main root polygon"]
    font_size_ = None
    linewidth_poly = 8
    linewidth_bbox = 5

    # Create a figure and a subplot
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(full_image, cmap='gray')
    ax[1].imshow(full_image, cmap='gray')
    ax[0].set_title("Synthetic root", size=font_size_)
    ax[1].set_title("Annotated Synthetic root", size=font_size_)

    root_poly_points = scale_up_polygon(root_polygon, full_image.shape)
    ax[1].plot(root_poly_points[:, 0], root_poly_points[:, 1], 'g-',
               linewidth=linewidth_poly, label='Main Root polygon')

    root_bbox = scale_up_bbox(root_bbox, full_image.shape)
    rect = patches.Rectangle((root_bbox[0], root_bbox[1]), root_bbox[2], root_bbox[3],
                             linewidth=linewidth_bbox, edgecolor='purple',
                             facecolor='none', label='Main Root BBox')
    ax[1].add_patch(rect)

    # Plot bounding boxes for hairs
    for bbox in hairs_bbox:
        hair_bbox = scale_up_bbox(bbox, full_image.shape)
        rect = patches.Rectangle((hair_bbox[0], hair_bbox[1]), hair_bbox[2], hair_bbox[3],
                                 linewidth=linewidth_bbox, edgecolor='r', facecolor='none')
        ax[1].add_patch(rect)

    # Plot polygons for hairs (each polygon is a sequence of x, y coordinates)
    for polygon in hairs_polygons:
        poly_points = scale_up_polygon(polygon, full_image.shape)
        ax[1].plot(poly_points[:, 0], poly_points[:, 1], color='blue',
                   linewidth=3)

    ax[0].set_axis_off()
    ax[1].set_axis_off()
    plt.show()


def add_annotations_to_coco(coco_dataset, image_id, properties):
    coco_dataset.add_annotation(
        image_id=image_id,
        category_id=1,  # Assuming category_id for root is 1
        bbox=properties["Main root bbox"],
        segmentation=[properties["Main root polygon"]],
        area=calculate_polygon_area(properties["Main root polygon"], image_shape=properties["full image"].shape)
    )

    for i in range(properties["hair count"]):
        coco_dataset.add_annotation(
            image_id=image_id,
            category_id=0,  # Assuming category_id for hair is 0
            bbox=properties["hairs bbox"][i],
            segmentation=[properties["hairs polygons"][i]],
            area=calculate_polygon_area(properties["hairs polygons"][i], image_shape=properties["full image"].shape)
        )


def annotate_specific_params_coco(coco, images_per_root, n_unique_roots, params):
    param_id = 0
    for main_root_points in generator_main_roots(n_unique_roots):
        root_image_class = RootImageGenerator(main_root_points, **params)

        for _ in range(images_per_root):
            properties = root_image_class.generate()
            full_path, _ = save_binary_image(properties["full image"], "dataset\\images", params, param_id,
                                             hair_count=properties["hair count"])
            im_id = coco.add_image(file_path=full_path, width=params["img_width"], height=params["img_height"])
            add_annotations_to_coco(coco, im_id, properties)
            param_id += 1


def run_coco(all_params, seed=None, plot=False):
    anno_per_root = 3
    roots_per_anno = 3
    iteration_num = count_iterations(all_params)

    coco = CocoDataset()
    coco.add_category(category_id=0, name="hair", supercategory="plant")
    coco.add_category(category_id=1, name="root", supercategory="plant")

    for params in tqdm(generator_combination_dict(all_params), total=iteration_num):
        annotate_specific_params_coco(coco, images_per_root=anno_per_root,
                                      n_unique_roots=roots_per_anno, params=params)
    coco.save_to_file('coco_format.json')


def save_annotation_yolo(prop_: dict, base_path: str, file_name: str):
    p = pathlib.Path(base_path)
    p.mkdir(parents=True, exist_ok=True)

    with open(f'{base_path}\\{file_name}', 'w') as f:
        row = f'1 {" ".join([str(val) for val in prop_["Main root polygon"]])}\n'
        f.write(row)

        for i in range(prop_["hair count"]):
            row = f'0 {" ".join([str(val) for val in prop_["hairs polygons"][i]])}\n'
            f.write(row)


def annotate_specific_params_yolo(images_per_root, n_unique_roots, params):
    rect_out_ = (params["img_width"]*0.1, params["img_height"]*0.1, params["img_width"]*0.9, params["img_height"]*0.9)
    for main_root_points in generator_main_roots(n_unique_roots, rect_out_):
        root_image_class = RootImageGenerator(main_root_points, **params)

        for _ in range(images_per_root):
            properties = root_image_class.generate(new_shape=(320, 320))
            _, filename = save_binary_image(properties["full image"], "dataset\\images", params,
                                            annotate_specific_params_yolo.counter_id,
                                            hair_count=properties["hair count"])
            filename_txt = filename.split('.')[0] + ".txt"
            save_annotation_yolo(properties, "dataset\\labels", filename_txt)
            annotate_specific_params_yolo.counter_id += 1


def run_yolo(all_params, n_main_root_=3, hair_gen_per_main_root_=3):
    iteration_num = count_iterations(all_params)

    annotate_specific_params_yolo.counter_id = 0

    for params in tqdm(generator_combination_dict(all_params), total=iteration_num):
        annotate_specific_params_yolo(images_per_root=hair_gen_per_main_root_,
                                      n_unique_roots=n_main_root_, params=params)


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
    N = 50

    params = {
        "root_width": 30,
        "root_width_std": 2,
        "hair_length": 50,
        "hair_length_std": 40,
        "hair_thickness": 5,
        "hair_thickness_std": 2,
        "hair_craziness": 0.90,  # 0 or 1
        "hair_density": 0.2,
        "img_width": 960,
        "img_height": 960,
        "root_start_percent": 0.20,
        "root_end_percent": 0.03,
        "hair_type": "random_walk"  # ["bezier", "random_walk-walk"]
    }
    # rect_out_ = (50, 50, 250, 250)
    rect_out_ = (params["img_width"]*0.1, params["img_height"]*0.1, params["img_width"]*0.9, params["img_height"]*0.9)

    for main_root_points in generator_main_roots(N, rect=rect_out_):
        root_image_class = RootImageGenerator(main_root_points, **params)
        properties = root_image_class.generate(add_soil=True, add_flare=True)
        plot_with_annotations(properties)


if __name__ == '__main__':

    possibilities = {
        "root_width": [20, 30, 40],
        "root_width_std": [2, 3],
        "hair_length": [20, 40, 60, 100],
        "hair_length_std": [5, 50],
        "hair_thickness": [5, 8],
        "hair_thickness_std": [2],
        "hair_craziness": [0.9, 0.98],  # 0 or 1
        "hair_density": [0.15, 0.20],
        "img_width": 960,
        "img_height": 960,
        "root_start_percent": [0.20],
        "root_end_percent": [0.03, 0.50],
        "hair_type": "random_walk"  # ["bezier", "random_walk-walk"]
    }
    n_main_root = 10
    hair_gen_per_main_root = 3
    print(f'Number of Images to generate: {count_iterations(possibilities) * n_main_root * hair_gen_per_main_root}')
    # run_coco(possibilities)  # doesnt seem to work right
    # show_images()
    run_yolo(possibilities, n_main_root, hair_gen_per_main_root)
