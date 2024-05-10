import time

import matplotlib.pyplot as plt
import numpy as np

from root_hair_counting import get_hairs_contours
from generate_main_root_points import generator_main_roots
from dataGeneration.gen_fake_images_v2 import RootImageGenerator


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

  smape = 100 * np.mean(2 * np.abs(predictions - ground_truth) / (np.abs(predictions) + np.abs(ground_truth) + smoothing_constant))
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
    "root_width_std": 0,
    "hair_length": 30,
    "hair_length_std": 10,
    "hair_thickness": 1,
    "hair_thickness_std": 0,
    "hair_craziness": 1,
    "hair_n": 50,
    "img_width": 300,
    "img_height": 300
}
hair_density = 0.08

gt_counts = []
pred_counts = []

print(f"generating {N} images...")
gen_count = 0
plot_images = 0
base = r'res/hair_count_alg_on_gen_images/'
plot_ = False
for main_root_points in generator_main_roots(N, rect_out, rect_in):

  root_length = len(main_root_points)
  if root_length < 2:
    continue
  gen_count += 1

  params["hair_n"] = int(root_length * hair_density)
  root_image_class = RootImageGenerator(main_root_points, **params)
  properties = root_image_class.generate()

  if True and gen_count % 100 == 0:
    file = f"{str(time.time()).split('.')[1]}.png"
    file_name = base + file
    print(gen_count)
  else:
    file_name = None

  hair_contours = get_hairs_contours(properties['full image'],
                                     truth_count=properties['hair count'], plot_=False,
                                     filename=file_name)
  # print(f"True {properties['hair count']}, estimated {len(hair_contours)} ")
  gt_counts.append(properties['hair count'])
  pred_counts.append(len(hair_contours))
print(f"Tested {gen_count} valid images")
print("Calculating counting Algorithm Error:")
calculate_metrics(pred_counts, gt_counts)
