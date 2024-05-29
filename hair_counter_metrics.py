import time

import numpy as np
from tqdm import tqdm
from image_processing_methods.root_hair_counting import get_hairs_contours
from SynDataGeneration.gen_main_root_points import generator_main_roots
from SynDataGeneration.gen_synthetic_images import RootImageGenerator


def calculate_metrics(predictions, ground_truth, smoothing_constant=1e-8):
  # Convert lists to numpy arrays
  predictions = np.array(predictions)
  ground_truth = np.array(ground_truth)

  # Calculating Mean Absolute Error (MAE)
  mae = np.mean(np.abs(predictions - ground_truth))
  print(f"Mean Absolute Error (MAE): {mae:.2f}")

  # Calculating Mean Squared Error (MSE)
  mse = np.mean((predictions - ground_truth) ** 2)
  print(f"Mean Squared Error (MSE): {mse:.2f}")

  # Calculating Root Mean Squared Error (RMSE)
  rmse = np.sqrt(mse)
  print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

  smape = 100 * np.mean(
    2 * np.abs(predictions - ground_truth) / (np.abs(predictions) + np.abs(ground_truth) + smoothing_constant))
  print(f"Symmetric Mean Absolute Percentage Error (SMAPE): {smape:.2f}%")

  num_zeros = np.count_nonzero(ground_truth == 0)
  print(f"Number of ground truth values that are zero: {num_zeros}")


"""
      "full image": merged_image,
      "only roots": root_image_bi,
      "only hairs": hairs_image_bi,
      "hair count": hair_num,
      "hairs polygons": hairs_poly,
      "hairs bbox": hairs_bbox
"""

np.random.seed(21)

rect_out = (50, 50, 250, 250)
delt = 20
rect_in = (rect_out[0] + delt, rect_out[1] + delt, rect_out[2] - delt, rect_out[3] - delt)

N = 1000
params = {
  "root_width": 5,
  "root_width_std": 1,
  "hair_length": 5,
  "hair_length_std": 10,
  "hair_thickness": 1,
  "hair_thickness_std": 0,
  "hair_craziness": 0,
  "hair_density": 0.12,
  "img_width": 300,
  "img_height": 300
}

gt_counts = []
pred_counts = []

gen_count = 0
plot_images = 0
base = r'res/hair_count_alg_on_gen_images/'
plot_ = True

print(f"generating {N} images...")
for main_root_points in tqdm(generator_main_roots(N)):

  root_length = len(main_root_points)
  if root_length < 2:
    continue
  gen_count += 1

  root_image_class = RootImageGenerator(main_root_points, **params)
  properties = root_image_class.generate()

  if gen_count % 100 == 0:
    file = f"{str(time.time()).split('.')[1]}.png"
    file_name = base + file
    plot_ = True
  else:
    plot_ = False
    file_name = None

  hair_contours = get_hairs_contours(properties['full image'],
                                     truth_count=properties['hair count'], plot_=plot_,
                                     filename=file_name)

  gt_counts.append(properties['hair count'])
  pred_counts.append(len(hair_contours))

print(f"Tested {gen_count} valid images")
print("Calculating counting Algorithm Error:")
calculate_metrics(pred_counts, gt_counts)
