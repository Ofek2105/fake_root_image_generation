import fiftyone as fo

coco_dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path="dataset/",
    labels_path="dataset/annotations.json",
    include_id=True
)

session = fo.launch_app(coco_dataset)

print("Press Ctrl+C to stop the session and exit.")
try:
    while True:
        pass
except KeyboardInterrupt:
    print("\nSession closed.")
