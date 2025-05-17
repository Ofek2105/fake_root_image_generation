import cv2
import os
import numpy as np

"""
A quick script that helps visualize yolo segmentation labels on the data for quick manual validating. 
"""

image_dir = 'datasetV1/images'
label_dir = 'datasetV1/labels'
ext = '.png'            # change to .png if needed

for label_file in os.listdir(label_dir):
    if not label_file.endswith('.txt'):
        continue

    image_file = label_file.replace('.txt', ext)
    image_path = os.path.join(image_dir, image_file)
    label_path = os.path.join(label_dir, label_file)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found: {image_path}")
        continue

    annotated = image.copy()
    h, w = image.shape[:2]

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))

            points = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * w)
                y = int(coords[i+1] * h)
                points.append([x, y])

            points = np.array([points], dtype=np.int32)
            color = (0, 255, 0)
            cv2.polylines(annotated, [points], isClosed=True, color=color, thickness=2)
            cv2.putText(annotated, str(cls), tuple(points[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    # Combine original and annotated images side-by-side
    combined = cv2.hconcat([image, annotated])
    cv2.imshow('Unannotated (left) | Annotated (right)', combined)

    key = cv2.waitKey(0)
    if key == 27:  # ESC to quit
        break

cv2.destroyAllWindows()
