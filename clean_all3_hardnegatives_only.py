import json

INPUT_FILE = "/home/frauas/ODT6/ObjectDetectionWithTransformersODT6/my_dataset/All-3.json"
OUTPUT_FILE = "/home/frauas/ODT6/ObjectDetectionWithTransformersODT6/my_dataset/All-3v1.json"

OLD_NO_OBJECT_ID = 2  # from your dataset

with open(INPUT_FILE, "r") as f:
    data = json.load(f)

# Filter annotations: keep only hard negatives
filtered_annotations = [
    ann for ann in data["annotations"]
    if ann["category_id"] == OLD_NO_OBJECT_ID
]

# Get image IDs that contain hard negatives
valid_image_ids = set(ann["image_id"] for ann in filtered_annotations)

# Filter images to keep only those with hard negatives
filtered_images = [
    img for img in data["images"]
    if img["id"] in valid_image_ids
]

# Reassign category_id to 0 (clean index)
for ann in filtered_annotations:
    ann["category_id"] = 0

# Replace dataset fields
clean_data = {
    "info": data.get("info", {}),
    "images": filtered_images,
    "annotations": filtered_annotations,
    "categories": [
        {
            "id": 0,
            "name": "no-object",
            "supercategory": None
        }
    ]
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(clean_data, f, indent=2)

print("Hard negative dataset saved to", OUTPUT_FILE)
print("Images kept:", len(filtered_images))
print("Annotations kept:", len(filtered_annotations))