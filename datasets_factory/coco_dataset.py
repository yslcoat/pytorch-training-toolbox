import pathlib
from collections import defaultdict
from collections.abc import Sequence
from PIL import Image, ImageDraw
from typing import Dict, List, Optional
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.ops import box_convert


import os
import pathlib
from PIL import Image
from typing import Dict, List
import json
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from torchvision.ops import box_convert


class CocoDataset(Dataset):
    def __init__(
        self,
        root_dir: pathlib.Path,
        partition: str = "train",
        convert_bbox_format: bool = True,
        transforms=None,
    ) -> None:
        self.root_dir = pathlib.Path(root_dir)
        self.convert_bbox_format = convert_bbox_format
        self.partition = partition
        self.transforms = transforms
        self.annotations_file = self.root_dir / "annotations" / f"instances_{partition}2017.json"
        self.images_dir = self.root_dir / f"{partition}2017"

        try:
            with open(self.annotations_file) as f:
                self.coco_data = json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading annotations: {e}")

        self.image_info = {img["id"]: img for img in self.coco_data.get("images", [])}
        self.image_ids = list(self.image_info.keys())

        self.img_to_anns = defaultdict(list)
        for ann in self.coco_data.get("annotations", []):
            self.img_to_anns[ann["image_id"]].append(ann)

    def __len__(self) -> int:
        return len(self.image_ids)

    def load_image(self, image_path: pathlib.Path) -> Image.Image:
        with Image.open(image_path) as image:
            return image.convert("RGB")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_id = self.image_ids[idx]
        image_meta = self.image_info[img_id]
        image_path = self.images_dir / image_meta["file_name"]
        image = self.load_image(image_path)

        anns: List[dict] = self.img_to_anns.get(img_id, [])
        if len(anns) > 0:
            boxes = torch.tensor([ann["bbox"] for ann in anns], dtype=torch.float32)
            if self.convert_bbox_format:
                boxes = box_convert(boxes, in_fmt="xywh", out_fmt="xyxy")
            labels = torch.tensor([ann.get("category_id", 0) for ann in anns], dtype=torch.int64)
            area = torch.tensor([ann.get("area", 0.0) for ann in anns], dtype=torch.float32)
            iscrowd = torch.tensor([ann.get("iscrowd", 0) for ann in anns], dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "area": area,
            "iscrowd": iscrowd,
            "image_id": torch.tensor([img_id], dtype=torch.int64),
            "orig_size": torch.tensor([image_meta["height"], image_meta["width"]], dtype=torch.int64),
            "size": torch.tensor([image_meta["height"], image_meta["width"]], dtype=torch.int64),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return {"image": image, "target": target}

