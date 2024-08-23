import time

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from image_processing_methods.root_hair_counting import get_hairs_contours
from SynDataGeneration.gen_main_root_points import generator_main_roots
from SynDataGeneration.gen_synthetic_images import RootImageGenerator
from create_dataset import plot_fake_and_pred_images, scale_up_polygon
import os
import cv2


def calculate_metrics(predictions, ground_truth, smoothing_window):
    # Convert lists to numpy arrays
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    medae = np.median(np.abs(predictions - ground_truth))
    print(f"Median Absolute Error (MedAE): {medae:.2f}")

    mae = np.mean(ground_truth - predictions)
    print(f"Mean Error gt-pred (MAE): {mae:.2f}")

    # Calculating Mean Absolute Error (MAE)
    mae = np.mean(np.abs(predictions - ground_truth))
    print(f"Mean Absolute Error (MAE): {mae:.2f}")

    # Calculating Mean Squared Error (MSE)
    mse = np.mean((predictions - ground_truth) ** 2)
    print(f"Mean Squared Error (MSE): {mse:.2f}")

    # Calculating Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    num_zeros = np.count_nonzero(ground_truth == 0)
    print(f"Number of ground truth values that are zero: {num_zeros}")

    if smoothing_window == 0:
        ground_truth = np.convolve(ground_truth, np.ones(smoothing_window) / smoothing_window, mode='valid')
        predictions = np.convolve(predictions, np.ones(smoothing_window) / smoothing_window, mode='valid')

    plt.plot(list(range(len(ground_truth))), ground_truth, 'g', label="Ground truth")
    plt.plot(list(range(len(predictions))), predictions, 'r', label="Predictions")
    plt.legend()
    plt.show()


"""
      "full image": merged_image,
      "only roots": root_image_bi,
      "only hairs": hairs_image_bi,
      "hair count": hair_num,
      "hairs polygons": hairs_poly,
      "hairs bbox": hairs_bbox
"""


def plot_image_and_annotations(image, segmentations_tuple, contours, file_path=None):
    hairs_segmentations, root_seg = segmentations_tuple

    n_plots = 2 if contours is None else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(15, 5))

    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    axes[0].imshow(image_rgb)
    axes[0].set_title("Base Image")
    axes[0].axis('off')

    axes[1].imshow(image_rgb)
    for seg in hairs_segmentations:
        hair_seg = [float(i) for i in seg]
        hair_seg = np.array(hair_seg).reshape((-1, 2))
        hair_seg_points = scale_up_polygon(hair_seg, image.shape)
        axes[1].plot(hair_seg_points[:, 0], hair_seg_points[:, 1], '-g', linewidth=1)

    root_seg = np.array(root_seg[0]).reshape((-1, 2))
    root_seg_points = scale_up_polygon(root_seg, image.shape)
    axes[1].plot(root_seg_points[:, 0], root_seg_points[:, 1], '-b', linewidth=3)

    axes[1].set_title(f"Image with Label Annotation ({len(hairs_segmentations)} hairs)")
    axes[1].axis('off')

    if contours is not None:
        contour_image = image_rgb.copy()  # Create an all-black image
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
        axes[2].imshow(contour_image)
        axes[2].set_title(f"Image with Contours ({len(contours)} hairs)")
        axes[2].axis('off')

    if file_path is not None:
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def count_zero_classes(label_image):
    return (label_image == 0).sum()


def read_bin_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    return binary_image


def get_prediction_metrics_from_dataset(dataset_path, save_folder_path=1, save_interval=3, max_images=None):
    images_folder = os.path.join(dataset_path, "images")
    labels_folder = os.path.join(dataset_path, "labels")

    im_count = 0
    gt_ = []
    preds_ = []
    for image_name in tqdm(os.listdir(images_folder)):
        im_count += 1

        image_path = os.path.join(images_folder, image_name)
        label_path = os.path.join(labels_folder, image_name[:-4]) + ".txt"

        image = read_bin_image(image_path)
        segmentations_tuple = get_seg_from_txt(label_path)
        pred_contours = get_hairs_contours(image)

        haris_num_gt = len(segmentations_tuple[0])
        gt_.append(haris_num_gt)
        preds_.append(len(pred_contours))

        if save_interval is not None and im_count % save_interval == 0:
            file_path = f"{save_folder_path}{str(time.time()).split('.')[1]}.png"
            plot_image_and_annotations(image, segmentations_tuple, None, file_path)

        if max_images is not None and im_count >= max_images:
            break
    return preds_, gt_


def get_seg_from_txt(label_path):
    segmentations_list_0 = []
    segmentations_list_1 = []

    with open(label_path, "r") as f:
        for line in f.readlines():
            obj_idx = line[0]
            seg_arr = np.array(line[2:].split(" "), dtype=np.float32)

            if obj_idx == "0":
                segmentations_list_0.append(seg_arr)  # start form 2 cause of index and space
            if obj_idx == "1":
                segmentations_list_1.append(seg_arr)  # start form 2 cause of index and space

    return segmentations_list_0, segmentations_list_1


def get_predictions_and_truth(save_folder_path, save_interval=3):
    """

  :param save_folder_path:
  :param save_interval: set the number of images between saving plot. set to -1 for no saving
  :return:
  """
    gt_counts = []
    pred_counts = []
    gen_count = 0

    print(f"generating {N} images...")
    for main_root_points in tqdm(generator_main_roots(N)):
        gen_count += 1

        root_image_class = RootImageGenerator(main_root_points, **params)
        properties = root_image_class.generate()

        hair_contours = get_hairs_contours(properties['full image'],
                                           truth_count=properties['hair count'], plot_=False,
                                           filename=None)

        if gen_count % save_interval == 0:
            file_path = f"{save_folder_path}{str(time.time()).split('.')[1]}.png"
            plot_fake_and_pred_images(fake_im_properties=properties, contours=hair_contours, file_path=file_path)

        gt_counts.append(properties['hair count'])
        pred_counts.append(len(hair_contours))
    return gt_counts, pred_counts


np.random.seed(21)

params = {
    "root_width": 5,
    "root_width_std": 0,
    "hair_length": 5,
    "hair_length_std": 10,
    "hair_thickness": 1,
    "hair_thickness_std": 0,
    "hair_craziness": 0,
    "hair_density": 0.12,
    "img_width": 300,
    "img_height": 300
}

# gt, preds = get_predictions_and_truth(save_folder_path=r'results/hair_counts_preds/')
preds, gt = get_prediction_metrics_from_dataset('dataset', save_folder_path='results/hair_counts_preds/',
                                                save_interval=100, max_images=None)
print(f"Tested {len(gt)} images")
print("Calculating counting Algorithm Error:")
calculate_metrics(preds, gt, smoothing_window=100)
