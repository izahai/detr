import torch
from collections import defaultdict
import os
from datasets.coco_eval import CocoEvaluator

import argparse
import json
from collections import defaultdict
from datasets import build_dataset, get_coco_api_from_dataset
from main import get_args_parser

def popo(gt_file, txt_file):
    # Load image_id mapping from gt.json
    with open(gt_file, "r") as f:
        gt_data = json.load(f)

    # Create mapping: "010_2.jpg" → image_id
    filename_to_id = {img["file_name"]: img["id"] for img in gt_data["images"]}

    # Prepare a mapping: (video, frame) → image_id
    vid_frame_to_id = {}
    for fname, img_id in filename_to_id.items():
        video, frame = fname.replace('.jpg', '').split('_')
        vid_frame_to_id[(int(video), int(frame))] = img_id

    # Initialize output dictionary
    res = defaultdict(lambda: {"boxes": [], "labels": [], "scores": []})

    # Read predictions from predict.txt
    with open(txt_file, "r") as f:
        for line in f:
            video, frame, x, y, w, h, class_id, score = line.strip().split(',')
            video, frame = int(video), int(frame)
            x, y, w, h = float(x), float(y), float(w), float(h)
            class_id, score = int(float(class_id)), float(score)

            image_id = vid_frame_to_id.get((video, frame))
            if image_id is None:
                continue  # skip unknown frames

            # Convert to [x_min, y_min, x_max, y_max]
            box = [x, y, x + w, y + h]
            res[image_id]["boxes"].append(box)
            res[image_id]["labels"].append(class_id)
            res[image_id]["scores"].append(score)

    # Convert lists to tensors
    for image_id in res:
        res[image_id]["boxes"] = torch.tensor(res[image_id]["boxes"])
        res[image_id]["labels"] = torch.tensor(res[image_id]["labels"])
        res[image_id]["scores"] = torch.tensor(res[image_id]["scores"])
    return res

def evaluate_txt_json(base_ds, iou_types, res):
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    coco_evaluator.update(res)

    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    
    return res

# Usage Example:
# Path to your ground truth and prediction .txt files
gt_file = 'data/aicity2024_track5_train/val.json'
#pred_file = 'all_1_100.txt'
pred_file = 'bb_txt/bb_012.txt'
#out_file = 'results/mAP_016.json'

parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()

dataset_val = build_dataset(image_set='val', args=args)
base_ds = get_coco_api_from_dataset(dataset_val)

evaluate_txt_json(base_ds, ['bbox'], popo(gt_file, pred_file))