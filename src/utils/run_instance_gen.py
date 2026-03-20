import wandb
import os
import time
import argparse, yaml
from pathlib import Path
import math
import sys
from tqdm import tqdm

import torch
from src.dataset import BUTIDDataset
# Add unisign path to system environment

from src.dataset import BUTIDDataset
from src.config import (
    train_label_paths,
    dev_label_paths,
    test_label_paths,
)

def main(args):

    train_data = BUTIDDataset(
        path=train_label_paths[args.dataset], args=args, phase="train"
    )
    print(train_data)
    print(train_data[5000])

    dev_data = BUTIDDataset(path=dev_label_paths[args.dataset], args=args, phase="dev")
    print(dev_data)
    test_data = BUTIDDataset(
        path=test_label_paths[args.dataset], args=args, phase="test"
    )
    print(test_data)

    os.makedirs(args.output_dir, exist_ok=True)
    # iterate over the dataset and save each instance .pt file
    for i in tqdm(range(len(train_data)), desc="Saving train instances"):
        video_id  = train_data.list_key[i]['video_id']
        caption_id = train_data.list_key[i]['key']
        if os.path.exists(os.path.join(args.output_dir, video_id, f"{caption_id}.pt")):
            continue
        key, pose_sample, vq_sample, text, gloss, task = train_data[i]
        if pose_sample is not None:
            video_id = key.split(".")[0]
            instance = {
                "key": key,
                "pose": pose_sample,
                "vq": vq_sample,
                "text": text,
                "gloss": gloss,
                "task": task,
            }
            os.makedirs(os.path.join(args.output_dir, video_id), exist_ok=True)
            torch.save(instance, os.path.join(args.output_dir, video_id, f"{caption_id}.pt"))
        
    for i in tqdm(range(len(dev_data)), desc="Saving dev instances"):
        video_id  = dev_data.list_key[i]['video_id']
        caption_id = dev_data.list_key[i]['key']
        if os.path.exists(os.path.join(args.output_dir, video_id, f"{caption_id}.pt")):
            continue
        key, pose_sample, vq_sample, text, gloss, task = dev_data[i]
        if pose_sample is not None:
            video_id = key.split(".")[0]
            instance = {
                "key": key,
                "pose": pose_sample,
                "vq": vq_sample,
                "text": text,
                "gloss": gloss,
                "task": task,
            }
            os.makedirs(os.path.join(args.output_dir, video_id), exist_ok=True)
            torch.save(instance, os.path.join(args.output_dir, video_id, f"{caption_id}.pt"))
        
    for i in tqdm(range(len(test_data)), desc="Saving test instances"):
        video_id  = test_data.list_key[i]['video_id']
        caption_id = test_data.list_key[i]['key']
        if os.path.exists(os.path.join(args.output_dir, video_id, f"{caption_id}.pt")):
            continue
        
        key, pose_sample, vq_sample, text, gloss, task = test_data[i]
        if pose_sample is not None:
            video_id = key.split(".")[0]
            instance = {
                "key": key,
                "pose": pose_sample,
                "vq": vq_sample,
                "text": text,
                "gloss": gloss,
                "task": task,
            }
            os.makedirs(os.path.join(args.output_dir, video_id), exist_ok=True)
            torch.save(instance, os.path.join(args.output_dir, video_id, f"{caption_id}.pt"))
    

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from src.args import get_args_parser

    parser = argparse.ArgumentParser("Uni-Sign scripts", parents=[get_args_parser()])

    parser.add_argument(
        "--config", default="", type=str, help="path to a YAML config file"
    )
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        # Set YAML values as defaults so CLI args take precedence
        parser.set_defaults(**cfg)
        args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
