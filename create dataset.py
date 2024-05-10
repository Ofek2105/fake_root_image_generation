import matplotlib.pyplot as plt
import numpy as np

from SynDataGeneration.gen_main_root_points import generator_main_roots
from SynDataGeneration.gen_synthetic_images import RootImageGenerator
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


N = 5
params = {
    "root_width": 5,
    "root_width_std": 1,
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


for main_root_points in generator_main_roots(N, rect_out, rect_in):
  root_length = len(main_root_points)
  params["hair_n"] = int(root_length * hair_density)
  root_image_class = RootImageGenerator(main_root_points, **params)
  properties = root_image_class.generate()

  # plt.imshow(properties["full image"], cmap='gray')
  # plt.axis('off')
  # plt.title('Bin image')
  # #
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

# coco_gen = CocoDataset()
#
# coco_gen.add_category(category_id=1, name="root", supercategory="plant")
# coco_gen.add_category(category_id=2, name="hair", supercategory="plant")
#
# coco_gen.save_to_file('coco_format.json')
