import json
import os
from datetime import datetime


class CocoDataset:
  def __init__(self, info=None, licenses=None):
    self.dataset = {
      "info": info if info else {
        "year": datetime.now().year,
        "version": "1.0",
        "description": "COCO Format Dataset",
        "contributor": "",
        "url": "",
        "date_created": datetime.now().strftime("%Y-%m-%d")
      },
      "licenses": licenses if licenses else [],
      "images": [],
      "annotations": [],
      "categories": []
    }
    self.image_counter = 0
    self.annotation_counter = 0

  def add_image(self, file_path, width, height):
    self.image_counter += 1
    file_name = os.path.basename(file_path)
    self.dataset['images'].append({
      "id": self.image_counter,
      "file_name": file_name,
      "width": width,
      "height": height
    })
    return self.image_counter

  def add_annotation(self, image_id, category_id, bbox, segmentation, area):
    self.annotation_counter += 1

    bbox = [float(x) for x in bbox]

    if isinstance(segmentation[0], float):  # If segmentation is a single list of floats
      segmentation = [segmentation]
    segmentation = [[float(coord) for coord in segment] for segment in segmentation]

    if bbox is [] or segmentation is []:
      raise ValueError("Bbox or segmentation are empty")

    self.dataset['annotations'].append({
      "id": self.annotation_counter,
      "image_id": image_id,
      "category_id": category_id,
      "bbox": bbox,  # [x, y, width, height]
      "segmentation": segmentation,  # [[x1, y1, x2, y2, ..., xn, yn]]
      "area": float(area),
      "iscrowd": 0
    })

  def add_category(self, category_id, name, supercategory):
    self.dataset['categories'].append({
      "id": category_id,
      "name": name,
      "supercategory": supercategory
    })

  def save_to_file(self, file_path):
    with open(file_path, 'w') as f:
      json.dump(self.dataset, f, indent=4)


if __name__ == '__main__':
  coco = CocoDataset()
  coco.add_category(category_id=1, name="root_hair", supercategory="plant")
  image_id = coco.add_image(file_path="path/to/image/00000001.jpg", width=1024, height=768)

  segmentations = [
    [[110, 200, 150, 200, 150, 280, 110, 280]],
    [[120, 210, 160, 210, 160, 290, 120, 290]],
    # Add more segmentations here
  ]

  bbox = [100, 200, 50, 80]  # Assuming the same bbox for simplicity
  category_id = 1

  for segmentation in segmentations:
    coco.add_annotation(image_id=image_id, category_id=category_id, bbox=bbox, segmentation=segmentation)

  coco.save_to_file('coco_format.json')
