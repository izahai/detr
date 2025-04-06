# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, evaluate_txt
from models import build_model
from mAP_txt import evaluate_txt_json, popo
from ensemble_boxes import *

import shutil

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')


    parser.add_argument('--change_csf_head', default=False)
    parser.add_argument('--build_train', default=False)
    # Inference
    parser.add_argument('--inference_a_video', default='', type=str)
    parser.add_argument('--train_val', default=False)

    return parser

def save_bb_txt(bb_res, eval_epoch, postprc="none"):
    file_path = f"bb_txt/bb_{eval_epoch:03}_{postprc}.txt"
    try:
        with open(file_path, "w") as f:
            for entry in bb_res:
                # Unpack the entry into individual variables
                video_id, frame_id, bbox_left, bbox_top, bbox_width, bbox_height, label, score = entry
                # Format the string as specified
                formatted_string = f"{video_id},{frame_id},{bbox_left:.17f},{bbox_top:.17f},{bbox_width:.17f},{bbox_height:.17f},{label},{score:.17f}\n"
                f.write(formatted_string)
        print(f"Saved bounding box data to {file_path}")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")

# UIT minority post-process
def count_samples_per_class(data):
    class_counts = [0,0,0,0,0,0,0,0,0] 
    for line in data:
        class_id = int(line[-2]) 
        class_counts[class_id-1] += 1
    return class_counts

def find_max(classes):
    classes_count = count_samples_per_class(classes)
    max_class = max(classes_count)
    return max_class, classes_count

def rm_bb_minoirty_in_a_video(res, minority_score):
    new_res = []
    for result in res:
        if result[-1] >= minority_score:
            new_res.append(result)
        #else:
         #   print(f"Remove {result[0]} {result[1]}")
    return new_res

def minority(classes, p=0.0001):
    n_maxclass, classes_count = find_max(classes)
    mean_samples = float(len(classes)/9)
    alpha = mean_samples/n_maxclass
    rare_classes = []
    for index, each_class in enumerate(classes_count):
        n_class = each_class
        if n_class < (n_maxclass * alpha):
            rare_classes.append(index)
    min_thresh = 1
    for each_class_index in rare_classes:
        for each_sample in classes:
            if each_class_index != int(each_sample[-2]-1):
                continue
            if each_sample[-1] < min_thresh:
                min_thresh = each_sample[-1]
    return max(min_thresh, p)

def multi_minority(bb_res):
    cur_video_id = 1
    bb_a_video = []
    new_results = []
    for instance_bb in bb_res:
        if instance_bb[0] == cur_video_id:
            bb_a_video.append(instance_bb)
        else:
            minority_score = minority(bb_a_video)
            #print(f"Minority score for video {cur_video_id}: {minority_score}")
            bb_a_video = rm_bb_minoirty_in_a_video(bb_a_video, minority_score)
            new_results.extend(bb_a_video)
            bb_a_video.clear()

            cur_video_id = instance_bb[0]
            bb_a_video.append(instance_bb)
    
    # Process the last video
    if bb_a_video:
        minority_score = minority(bb_a_video)
        #print(f"Minority score for video {cur_video_id}: {minority_score}")
        bb_a_video = rm_bb_minoirty_in_a_video(bb_a_video, minority_score)
        new_results.extend(bb_a_video)

    return new_results

def fuse(  # NOTE: fuse a single video!
    process_video_results: list,
    video_path: str='',
    iou_thr: float = 0.7, # default values of repo
    skip_box_thr: float = 0.0001, # default values of repo
) -> list:
    datas = process_video_results # list of [int(video_id), int(frame_id), bbox_left, bbox_top, bbox_width, bbox_height, float(label), score]
    results = []
    w, h = 1280, 720

    video_id = datas[0][0]
    cur_frame_id = datas[0][1]
    frame_bb = []

    for bb_ins in datas:
        frame_id = bb_ins[1]

        if frame_id != cur_frame_id:
            if frame_bb:
                boxes_list = []
                scores_list = []
                labels_list = []
                weights = [1] * len(frame_bb)

                for box in frame_bb:
                    x_min = box[2] / w
                    y_min = box[3] / h
                    x_max = (box[2] + box[4]) / w
                    y_max = (box[3] + box[5]) / h
                    boxes_list.append([x_min, y_min, x_max, y_max])
                    scores_list.append(box[7])
                    labels_list.append(box[6])

                fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
                    boxes_list, scores_list, labels_list, weights=weights,
                    iou_thr=iou_thr, skip_box_thr=skip_box_thr
                )

                for i in range(len(fused_boxes)):
                    x_min = fused_boxes[i][0] * w
                    y_min = fused_boxes[i][1] * h
                    width = (fused_boxes[i][2] - fused_boxes[i][0]) * w
                    height = (fused_boxes[i][3] - fused_boxes[i][1]) * h
                    results.append([video_id, cur_frame_id, x_min, y_min, width, height, fused_labels[i], fused_scores[i]])

            cur_frame_id = frame_id
            frame_bb = []

        frame_bb.append(bb_ins)

    if frame_bb:
        boxes_list = []
        scores_list = []
        labels_list = []
        weights = [1] * len(frame_bb)

        for box in frame_bb:
            x_min = box[2] / w
            y_min = box[3] / h
            x_max = (box[2] + box[4]) / w
            y_max = (box[3] + box[5]) / h
            boxes_list.append([x_min, y_min, x_max, y_max])
            scores_list.append(box[7])
            labels_list.append(box[6])

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list, weights=weights,
            iou_thr=iou_thr, skip_box_thr=skip_box_thr
        )

        for i in range(len(fused_boxes)):
            x_min = fused_boxes[i][0] * w
            y_min = fused_boxes[i][1] * h
            width = (fused_boxes[i][2] - fused_boxes[i][0]) * w
            height = (fused_boxes[i][3] - fused_boxes[i][1]) * h
            results.append([video_id, cur_frame_id, x_min, y_min, width, height, fused_labels[i], fused_scores[i]])

    return results

def multi_fuse(bb_res):
    new_res = []

    return new_res         

def detection_test_set(
    model, criterion, postprocessors,
    data_loader_val, base_ds, device, args
) -> list:
    
    bb_res, coco_evaluator = evaluate_txt(model, criterion, postprocessors,
        data_loader_val, base_ds, device, args.output_dir)
    
    match = re.search(r'.*checkpoint(\d+)\.pth', args.resume)
    if match:
        eval_epoch = int(match.group(1))

    full_metrics = {}
    for iou_type, coco_eval in coco_evaluator.coco_eval.items():
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

    # Save full metrics to a JSON file
    full_metrics_path = f"results/full_metrics_epoch{eval_epoch:03}.json"
    with open(full_metrics_path, "w") as f:
        json.dump(full_metrics, f, indent=2)
    print(f"Full evaluation metrics saved to {full_metrics_path}")
    
    return bb_res

def load_bb_txt(file_path):
    bb_res = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                # Parse the line into components
                values = line.strip().split(",")
                # Convert numeric values to appropriate types
                parsed_line = [
                    int(values[0]),  # video_id
                    int(values[1]),  # frame_id
                    float(values[2]),  # bbox_left
                    float(values[3]),  # bbox_top
                    float(values[4]),  # bbox_width
                    float(values[5]),  # bbox_height
                    int(float(values[6])),  # class_id
                    float(values[7])  # score
                ]
                bb_res.append(parsed_line)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
    
    return bb_res

def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)


    #if args.build_train:
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    base_ds = get_coco_api_from_dataset(dataset_val)
    output_dir = Path(args.output_dir)
        
    # load checkpoint
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    model_without_ddp.load_state_dict(checkpoint['model'])
    gt_file = 'data/aicity2024_track5_train/val.json'
 
    total_boxes = 0
    eval_epoch = 0
    for img_id in base_ds.imgs.keys():
        ann_ids = base_ds.getAnnIds(imgIds=img_id)
        anns = base_ds.loadAnns(ann_ids)
        total_boxes += len(anns)
    print(f"Total number of bounding boxes in the validation set: {total_boxes}")
    
    match = re.search(r'.*checkpoint(\d+)\.pth', args.resume)
    if match:
        eval_epoch = int(match.group(1))
    
    postprc='none'
    #ori_res = detection_test_set(model, criterion, postprocessors, data_loader_val, base_ds, device, args)
    ori_res = load_bb_txt(f"bb_txt/bb_{eval_epoch:03}_{postprc}.txt")
    save_bb_txt(ori_res, eval_epoch) # Save none postprocess bb  
    evaluate_txt_json(base_ds, ['bbox'], gt_file, f"bb_txt/bb_{eval_epoch:03}_{postprc}.txt")
     
    #postprc = 'fuse'
    #new_res = fuse(ori_res)
    #save_bb_txt(new_res, eval_epoch, postprc)
    #evaluate_txt_json(base_ds, ['bbox'], gt_file, f"bb_txt/bb_{eval_epoch:03}_{postprc}.txt")


    postprc = 'minority' 
    new_res = multi_minority(ori_res)
    save_bb_txt(new_res, eval_epoch, postprc)
    evaluate_txt_json(base_ds, ['bbox'], gt_file, f"bb_txt/bb_{eval_epoch:03}_{postprc}.txt")
    
    return
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
