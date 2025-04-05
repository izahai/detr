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
from engine import evaluate, train_one_epoch, inference_a_image
from models import build_model

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


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
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
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
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

    if args.distributed: # For multi-GPU training
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    elif not args.inference_a_video:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    print("output_dir: ", output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            args.change_csf_head = False
        
        

    if args.change_csf_head:
        num_classes = 10
        model_dict = model_without_ddp.state_dict()

        # Remove classification head (it has a different shape)
        checkpoint_model_dict = {k: v for k, v in checkpoint['model'].items() if "class_embed" not in k}

        # Load the remaining weights
        model_dict.update(checkpoint_model_dict)
        model_without_ddp.load_state_dict(model_dict)

        # Reinitialize class_embed layer for 11 classes (your dataset)
        hidden_dim = model_without_ddp.class_embed.in_features  # Should be 256

        model_without_ddp.class_embed = torch.nn.Linear(hidden_dim, num_classes + 1).to(device)  # +1 for background
    else:
        model_without_ddp.load_state_dict(checkpoint['model'])

 
    if args.inference_a_video:
        inference_a_image(
            model, postprocessors, args.inference_a_video, device, args.output_dir)
        return
    
    # if args.eval:
    #     test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
    #                                          data_loader_val, base_ds, device, args.output_dir)
    #     if args.output_dir:
    #         utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
    #     return
    
    if args.eval:
        total_boxes = 0
        for img_id in base_ds.imgs.keys():
            ann_ids = base_ds.getAnnIds(imgIds=img_id)
            anns = base_ds.loadAnns(ann_ids)
            total_boxes += len(anns)
        print(f"Total number of bounding boxes in the validation set: {total_boxes}")
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                            data_loader_val, base_ds, device, args.output_dir)
    
        if args.output_dir:
            # Try to extract epoch number from checkpoint filename
            match = re.search(r'.*checkpoint(\d+)\.pth', args.resume)
            if match:
                eval_epoch = int(match.group(1))
            else:
                eval_epoch = 0  # fallback if no epoch found in filename

            #eval_path = output_dir / f"eval_epoch{eval_epoch:03}.pth"
            #utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, eval_path)

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
            full_metrics_path = output_dir / f"full_metrics_epoch{eval_epoch:03}.json"
            with open(full_metrics_path, "w") as f:
                json.dump(full_metrics, f, indent=2)
            print(f"Full evaluation metrics saved to {full_metrics_path}")
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()

        if args.output_dir:
            # Saving checkpoints
            checkpoint_path = output_dir / f'checkpoint{epoch:03}.pth'  # Unique checkpoint per epoch
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)

        test_stats, coco_evaluator = {}, {}
        if args.train_val:
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
            )

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
            full_metrics_path =  f"full_metrics_epoch{eval_epoch:03}.json"
            with open(full_metrics_path, "w") as f:
                json.dump(full_metrics, f, indent=2)
            print(f"Full evaluation metrics saved to {full_metrics_path}")

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")


        shutil.copy(f"detr_train/checkpoints/checkpoint{epoch:03}.pth", "/content/drive/MyDrive")
        shutil.copy(f"full_metrics_epoch{epoch:03}.json", "/content/drive/MyDrive")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
