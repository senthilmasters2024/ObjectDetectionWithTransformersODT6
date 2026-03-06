"""
Clean All-3.json  →  All-3_clean_bbox_only.json

Keeps only:
  - Annotations with category_id == 2  ("No Object" / hard negatives)
  - Annotations that have a valid bounding box  (width > 0 and height > 0)

Strips segmentation polygons — training only needs bbox.
Remaps category_id to 0 and renames to "no-object".
"""

import json

INPUT_FILE  = "./my_dataset/All-3.json"
OUTPUT_FILE = "./my_dataset/All-3_clean_bbox_only.json"

NO_OBJECT_CATEGORY_ID = 2  # "No Object" in All-3.json

with open(INPUT_FILE, "r") as f:
    data = json.load(f)

total_annotations = len(data["annotations"])

# ── Step 1: keep only No Object annotations with valid bboxes ─────────────────
filtered_annotations = []
skipped_wrong_class = 0
skipped_bad_bbox = 0

for ann in data["annotations"]:
    if ann["category_id"] != NO_OBJECT_CATEGORY_ID:
        skipped_wrong_class += 1
        continue

    bbox = ann.get("bbox", [])
    if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
        skipped_bad_bbox += 1
        continue

    filtered_annotations.append(ann)

# ── Step 2: strip segmentation, remap category_id ────────────────────────────
clean_annotations = []
for i, ann in enumerate(filtered_annotations):
    clean_annotations.append({
        "id":          i + 1,
        "image_id":    ann["image_id"],
        "category_id": 0,           # remapped to 0
        "bbox":        ann["bbox"],
        "area":        ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
        "iscrowd":     ann.get("iscrowd", 0),
    })

# ── Step 3: keep only images referenced by the filtered annotations ───────────
valid_image_ids = set(ann["image_id"] for ann in clean_annotations)
filtered_images = [img for img in data["images"] if img["id"] in valid_image_ids]

# ── Step 4: write output ──────────────────────────────────────────────────────
output = {
    "info":        data.get("info", {}),
    "images":      filtered_images,
    "annotations": clean_annotations,
    "categories": [
        {"id": 0, "name": "no-object", "supercategory": None}
    ],
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"Input  : {INPUT_FILE}")
print(f"Output : {OUTPUT_FILE}")
print(f"")
print(f"Total annotations in input  : {total_annotations}")
print(f"  Skipped (wrong class)     : {skipped_wrong_class}")
print(f"  Skipped (invalid bbox)    : {skipped_bad_bbox}")
print(f"  Kept (no-object + valid)  : {len(clean_annotations)}")
print(f"")
print(f"Images kept : {len(filtered_images)}")
print(f"Done -> {OUTPUT_FILE}")
