import os
import json
import numpy as np
import cv2
from typing import List, Union


class COCODatasetGenerator:
    def __init__(self, output_dir: str):
        """
    Initialize COCO dataset generator

    Args:
        output_dir (str): Directory to save COCO format annotations
    """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.coco_dataset = {
            "info": {
                "year": 2024,
                "version": "1.0",
                "description": "Instance Segmentation Dataset",
                "contributor": "",
                "url": "",
                "date_created": ""
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "root", "supercategory": ""},
                {"id": 0, "name": "hair", "supercategory": ""}
            ]
        }

        self.image_id = 0
        self.annotation_id = 0

    def add_image(self, image_path: str, height: int, width: int):
        """
    Add image metadata to the COCO dataset

    Args:
        image_path (str): Path to the image file
        height (int): Image height
        width (int): Image width
    """
        self.image_id += 1
        image_info = {
            "id": self.image_id,
            "width": width,
            "height": height,
            "file_name": os.path.basename(image_path)
        }
        self.coco_dataset["images"].append(image_info)
        return self.image_id

    def add_annotation(
            self,
            image_id: int,
            segmentation: Union[List[float], np.ndarray],
            category_id: int
    ):
        """
    Add segmentation annotation to the COCO dataset

    Args:
        image_id (int): ID of the corresponding image
        segmentation (Union[List[float], np.ndarray]): Segmentation mask or polygon
        category_id (int): Category ID (1 for root, 2 for hair)
    """
        self.annotation_id += 1

        # Calculate bbox
        points = np.array(segmentation).reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(points)
        bbox = [float(x), float(y), float(w), float(h)]

        # Calculate area
        area = cv2.contourArea(points)

        annotation = {
            "id": self.annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": [segmentation],  # COCO format requires list of lists
            "area": float(area),
            "bbox": bbox,
            "iscrowd": 0
        }

        self.coco_dataset["annotations"].append(annotation)
        return self.annotation_id

    def save_annotations(self, filename: str = "annotations.json"):
        """
    Save COCO format annotations to a JSON file

    Args:
        filename (str): Output filename for annotations
    """
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(self.coco_dataset, f)
        print(f"Annotations saved to {output_path}")


# Example usage
def example_usage():
    # Initialize the generator
    generator = COCODatasetGenerator("/path/to/output")

    # Add an image
    image_path = "/path/to/image.png"
    image = cv2.imread(image_path)
    image_id = generator.add_image(
        image_path,
        height=image.shape[0],
        width=image.shape[1]
    )

    # Add root segmentation (polygon or mask)
    root_mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    generator.add_annotation(
        image_id,
        segmentation=root_mask,
        category_id=1  # root
    )

    # Add hair segmentations
    hair_masks = []  # Your list of hair masks
    for hair_mask in hair_masks:
        generator.add_annotation(
            image_id,
            segmentation=hair_mask,
            category_id=2  # hair
        )

    # Save annotations
    generator.save_annotations()