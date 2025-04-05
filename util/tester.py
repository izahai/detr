import os
import json
import argparse
from tqdm import tqdm

# Define your class labels
labels = ["motorbike", "DHelmet", "DNoHelmet", "P1Helmet", "P1NoHelmet", "P2Helmet", "P2NoHelmet", "P0Helmet", "P0NoHelmet"]
label_to_id = {name: i+1 for i, name in enumerate(labels)}  # COCO uses 1-indexed category IDs

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Convert YOLO-format labels to COCO format JSON')
parser.add_argument('--basepath', type=str, default="data/aicity2024_track5_train/", help='Base path of the dataset')
parser.add_argument('--outpath', type=str, default="coco_annotations.json", help='Output JSON file name')
parser.add_argument('--image_width', type=int, default=1280, help='Image width')
parser.add_argument('--image_height', type=int, default=720, help='Image height')
args = parser.parse_args()

# Define paths
basepath = args.basepath
all_txt_path = os.path.join(basepath, "gt")
outjson_path = os.path.join(basepath, args.outpath)

# Initialize COCO structure
images = []
annotations = []
categories = [{"id": i+1, "name": name} for i, name in enumerate(labels)]

annotation_id = 1
image_id = 1

width, height = args.image_width, args.image_height

uni_set = set()


with open("all_1_100.txt", 'r') as f:
    lines = f.readlines()
    uni_set = set()
    for line in lines:
        parts = line.strip().split(',')
        uni_set.add((parts[0], parts[1]))


#323053
# for tp in uni_set:
#     intt1 = int(tp[0])
#     intt2 = int(tp[1])
#     fname = f"{intt1:03}_{intt2}.txt"

#Iterate through all YOLO txt files 323639
for fname in tqdm(sorted(os.listdir(all_txt_path))):
    if not fname.endswith('.txt'):
        continue

    txt_file = os.path.join(all_txt_path, fname)
    image_name = fname.replace(".txt", ".jpg")

    # Register the image
    images.append({
        "id": image_id,
        "file_name": image_name,
        "width": args.image_width,
        "height": args.image_height
    })

    with open(txt_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        cls_id, x, y, w, h = map(float, parts)
        category_id = int(cls_id) + 1  # Adjust class index to 1-based for COCO
        x_min, y_min = int((x - w / 2) * width), int((y - h / 2) * height)
        x_max, y_max = int((x + w / 2) * width), int((y + h / 2) * height)

        annotations.append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
            "area": (x_max - x_min) * (y_max - y_min),
            "iscrowd": 0
        })
        annotation_id += 1

    image_id += 1

# Final COCO-style dictionary
coco_dict = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}

# Save as JSON
with open(outjson_path, 'w', encoding='utf-8') as f:
    json.dump(coco_dict, f, indent=2)

print(f"\n✅ COCO annotation file saved to: {outjson_path}")
print(f"Num of frames {len(coco_dict['images'])}")