import os
import pathlib
from PIL import Image
from typing import Tuple, Dict, List
import xml.etree.ElementTree as ET
import json

import torch
from torch.utils.data import Dataset
import torchvision


class CocoDataset(Dataset):
    def __init__(
        self,
        root_dir: pathlib.Path,
        partition: str = "train",
        transforms=None,
    ) -> None:
        self.annotations_file = pathlib.Path(root_dir, "annotations", f"instances_{partition}.json")
        try:
            with open(self.annotations_file) as f:
                self.annotations = json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading annotations: {e}")

        self.image_paths = [pathlib.Path(root_dir, "images", f"{img['file_name']}") for img in self.annotations["images"]] # Or alternatively list all image files in the directory, idk whats best
        self.image_ids = [img["id"] for img in self.annotations["images"]]
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("CocoDataset is not implemented yet")