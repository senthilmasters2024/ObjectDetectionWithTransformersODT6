import json
import os

INPUT_JSON = "/home/frauas/ODT6/ObjectDetectionWithTransformersODT6/my_dataset/All-3v1.json"
OUTPUT_JSON = "/home/frauas/ODT6/ObjectDetectionWithTransformersODT6/my_dataset/All-3_clean_bbox_only.json"

print("Loading:", INPUT_JSON)

with open(INPUT_JSON, "r") as f:
    data = json.load(f)

clean_annotations = []
valid_image_ids = set()

for ann in data.get("annotations", []):
    
    bbox = ann.get("bbox", [])
    
    # Keep only proper bounding boxes
    if (
        isinstance(bbox, list) and
        len(bbox) == 4 and
        bbox[2] > 0 and
        bbox[3] > 0
    ):
        ann["category_id"] = 0  # force no-object class
        ann["segmentation"] = []  # remove segmentation if any
        ann["area"] = bbox[2] * bbox[3]
        
        clean_annotations.append(ann)
        valid_image_ids.add(ann["image_id"])

print("Valid bbox annotations kept:", len(clean_annotations))

# Keep only images that have valid bbox annotations
clean_images = [
    img for img in data.get("images", [])
    if img["id"] in valid_image_ids
]

print("Images kept:", len(clean_images))

# Build clean COCO structure
clean_data = {
    "images": clean_images,
    "annotations": clean_annotations,
    "categories": [
        {
            "id": 0,
            "name": "no-object"
        }
    ]
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(clean_data, f, indent=2)

print("\nClean file saved as:", OUTPUT_JSON)
print("Done.")