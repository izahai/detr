# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    num_frames_with_boxes = 0
    total_predicted_boxes = 0  # Initialize counter for total predicted bounding boxes
    
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        total_predicted_boxes += sum(len(r['boxes']) for r in results)
        num_frames_with_boxes += sum(len(r['boxes']) > 0 for r in results)

        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    print("Number of frames having bouding box:", num_frames_with_boxes)
    print(f"Total number of predicted bounding boxes: {total_predicted_boxes}")
    return stats, coco_evaluator

import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_image(image_path):
    transform = T.Compose([
        T.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0), image

def draw_bboxes(image, boxes, scores, labels, class_names=None):
    """
    Draw bounding boxes on the image, displaying confidence scores and class indices.
    
    Args:
    - image: The image on which to draw the boxes.
    - boxes: Bounding boxes as a numpy array of shape (N, 4).
    - scores: Confidence scores corresponding to each box.
    - labels: Class indices corresponding to each box.
    - class_names: A list of class names (optional), where the index corresponds to the label.
    """
    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.imshow(image)
    
    for box, score, label in zip(boxes, scores, labels):
        x_min, y_min, x_max, y_max = box
        width, height = x_max - x_min, y_max - y_min
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        # Display score and label (class index)
        label_text = f'Class: {label}'
        score_text = f'{score:.2f}'
        
        # If class names are provided, display the class name instead of the index
        if class_names is not None:
            label_text = f'{class_names[label]}: {score:.2f}'
        
        ax.text(x_min, y_min, f'{label_text} {score_text}', color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))
    
    plt.show()

@torch.no_grad()
def inference_a_image(model, postprocessor, image_path, device, output_dir, confidence_threshold=0.5):
    """
    Perform inference on a single image and save the results.
    Only draws bounding boxes with confidence scores above the given threshold.
    Displays the class index along with the bounding box and confidence score.
    """
    model.eval()
    image_tensor, image = load_image(image_path)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        # Run the model
        outputs = model(image_tensor)
    
    # Get the original size
    orig_size = torch.tensor([[image.height, image.width]], device=device)
    print("orig_size", orig_size)
    print("orig_size_shape", orig_size.shape)
    print("PostProcessor:", postprocessor)

    # Access the 'bbox' postprocessor for bounding boxes
    postprocessor_bbox = postprocessor['bbox']
    
    # Post-process the outputs with the original size
    results = postprocessor_bbox(outputs, orig_size)[0]  # Get the first (and only) image's results
    
    # Extract boxes, scores, and labels (class indices)
    boxes = results['boxes'].cpu().numpy()
    scores = results['scores'].cpu().numpy()
    labels = results['labels'].cpu().numpy()  # Class indices for the bounding boxes
    
    # Filter out boxes with scores below the threshold
    confident_boxes = boxes[scores > confidence_threshold]
    confident_scores = scores[scores > confidence_threshold]
    confident_labels = labels[scores > confidence_threshold]  # Only keep the labels corresponding to confident boxes
    
    # Draw the filtered bounding boxes along with class indices and confidence scores
    draw_bboxes(image, confident_boxes, confident_scores, confident_labels)
    print(confident_boxes)

    # Optionally save the output to the specified directory (if needed)
    # output_path = os.path.join(output_dir, 'inference_result.png')
    # plt.savefig(output_path)

def get_filename_of_imaged_id(image_id, base_ds):
    if image_id in base_ds.imgs:
        return base_ds.imgs[image_id]['file_name']
    else:
        raise ValueError(f"Image ID {image_id} not found in the dataset.")

@torch.no_grad()
def evaluate_txt(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    num_frames_with_boxes = 0
    total_predicted_boxes = 0  # Initialize counter for total predicted bounding boxes
    uit_txt_format = [] # list of []

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        total_predicted_boxes += sum(len(r['boxes']) for r in results)
        num_frames_with_boxes += sum(len(r['boxes']) > 0 for r in results)

        # Process predictions and format them
        for target, result in zip(targets, results):
            image_id = target['image_id'].item()
            file_name = get_filename_of_imaged_id(image_id, base_ds)  # Get the filename
            video_id, frame_id = map(int, file_name.split('.')[0].split('_'))  # Extract video_id and frame_id

            boxes = result['boxes'].cpu().numpy()  # Bounding boxes
            scores = result['scores'].cpu().numpy()  # Confidence scores
            labels = result['labels'].cpu().numpy()  # Class labels

            for box, score, label in zip(boxes, scores, labels):
                bbox_left, bbox_top, bbox_right, bbox_bottom = box
                bbox_width = bbox_right - bbox_left
                bbox_height = bbox_bottom - bbox_top

                # Format the string
                formatted_string = f"{video_id},{frame_id},{bbox_left:.17f},{bbox_top:.17f},{bbox_width:.17f},{bbox_height:.17f},{label},{score:.17f}\n"
                uit_txt_format.append(formatted_string)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()

    print("Number of frames having bouding box:", num_frames_with_boxes)
    print(f"Total number of predicted bounding boxes: {total_predicted_boxes}")
    
    return uit_txt_format, coco_evaluator
