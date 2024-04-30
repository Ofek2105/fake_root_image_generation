import json


class CocoDataset:
  def __init__(self):
    self.dataset = {
      "info": {},
      "licenses": [],
      "images": [],
      "annotations": [],
      "categories": []
    }

  def add_image(self, image_id, file_name, width, height):
    self.dataset['images'].append({
      "id": image_id,
      "file_name": file_name,
      "width": width,
      "height": height
    })

  def add_annotation(self, anno_id, image_id, category_id, bbox, segmentation):
    self.dataset['annotations'].append({
      "id": anno_id,
      "image_id": image_id,
      "category_id": category_id,
      "bbox": bbox,  # [x, y, width, height]
      "segmentation": segmentation,  # [[x1, y1, x2, y2, ..., xn, yn]]
      "area": bbox[2] * bbox[3],  # width * height
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
  # Usage example
  coco = CocoDataset()
  coco.add_image(image_id=1, file_name="00000001.jpg", width=1024, height=768)
  coco.add_annotation(anno_id=1, image_id=1, category_id=1, bbox=[100, 200, 50, 80],
                      segmentation=[[110, 200, 150, 200, 150, 280, 110, 280]])
  coco.add_category(category_id=1, name="root_hair", supercategory="plant")
  coco.save_to_file('coco_format.json')
