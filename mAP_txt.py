import torch
from collections import defaultdict
import os
from datasets.coco_eval import CocoEvaluator

import argparse
import json
from collections import defaultdict
from datasets import build_dataset, get_coco_api_from_dataset
from main import get_args_parser
import re

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

import json
import numpy as np

def save_eval_json(coco_evaluator, tg_file):
    full_metrics = {}
    
    for iou_type, coco_eval in coco_evaluator.coco_eval.items():
        class_ap50 = {}
        
        if hasattr(coco_eval, "stats"):
            full_metrics[iou_type] = {
                "AP (IoU=0.50:0.95 | area=all | maxDets=100)": coco_eval.stats[0],
                "AP (IoU=0.50      | area=all | maxDets=100)": coco_eval.stats[1],
                "AP (IoU=0.75      | area=all | maxDets=100)": coco_eval.stats[2],
                "AP (IoU=0.50:0.95 | area=small | maxDets=100)": coco_eval.stats[3],
                "AP (IoU=0.50:0.95 | area=medium | maxDets=100)": coco_eval.stats[4],
                "AP (IoU=0.50:0.95 | area=large | maxDets=100)": coco_eval.stats[5],
                "AR (IoU=0.50:0.95 | area=all | maxDets=1)": coco_eval.stats[6],
                "AR (IoU=0.50:0.95 | area=all | maxDets=10)": coco_eval.stats[7],
                "AR (IoU=0.50:0.95 | area=all | maxDets=100)": coco_eval.stats[8],
                "AR (IoU=0.50:0.95 | area=small | maxDets=100)": coco_eval.stats[9],
                "AR (IoU=0.50:0.95 | area=medium | maxDets=100)": coco_eval.stats[10],
                "AR (IoU=0.50:0.95 | area=large | maxDets=100)": coco_eval.stats[11],
            }

        # Compute AP@50 for each class
        if coco_eval.eval is not None and "precision" in coco_eval.eval:
            precisions = coco_eval.eval["precision"]  # shape: [IoU, Recall, Class, Area, MaxDets]
            iou_thresh_idx = 0  # Index for IoU=0.50
            area_idx = 0        # Index for area='all'
            max_det_idx = 2     # Index for maxDets=100
            
            for idx, cat_id in enumerate(coco_eval.params.catIds):
                # Get precision values for this class at IoU=0.50
                class_precisions = precisions[iou_thresh_idx, :, idx, area_idx, max_det_idx]
                valid_precisions = class_precisions[class_precisions > -1]
                if valid_precisions.size > 0:
                    ap50 = np.mean(valid_precisions)
                    class_ap50[str(cat_id)] = round(float(ap50), 4)
                else:
                    class_ap50[str(cat_id)] = None
        
        if class_ap50:
            print("Per-class AP@50:")
            for cat_id, ap in class_ap50.items():
                print(f"  Class {cat_id}: AP@50 = {ap}")
            full_metrics[iou_type]["Per-class AP@50"] = class_ap50

    with open(tg_file, "w") as f:
        json.dump(full_metrics, f, indent=2)
    return

def evaluate_txt_json(base_ds, iou_types, gt_file, pred_file=str):
    res = popo(gt_file, pred_file)
    
    tg_file = pred_file.replace(".txt", ".json").replace("bb_txt/", "results/")

    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    coco_evaluator.update(res)

    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    
    save_eval_json(coco_evaluator, tg_file)
    return res

if __name__ == '__main__':
    # Usage Example:
    # Path to your ground truth and prediction .txt files
    gt_file = 'data/aicity2024_track5_train/val.json'
    pred_file = 'all_1_102.txt'
    #pred_file = 'bb_txt/bb_012.txt'
    #pred_file='final_results_conf_0.3_adaptive.txt'

    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    dataset_val = build_dataset(image_set='val', args=args)
    base_ds = get_coco_api_from_dataset(dataset_val)

    evaluate_txt_json(base_ds, ['bbox'], gt_file, pred_file)