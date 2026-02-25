import os
import pathlib
import logging
from PIL import Image
from typing import Tuple, Dict, List
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset
import torchvision


class ImageNetDataset(Dataset):
    """
    Custom imagenet dataset class. Expects the following file structure

    root_dir/
    ├── ILSVRC/
    │   ├── Data/
    │   │   └── CLS-LOC/
    │   │       ├── train/
    │   │       │   └── <class_id>/
    │   │       │       └── <filename>.JPEG
    │   │       └── val/
    │   │           └── <class_id>/
    │   │               └── <filename>.JPEG
    │   └── Annotations/
    │       └── CLS-LOC/
    │           ├── train/
    │           │   └── <class_id>/
    │           │       └── <filename>.xml
    │           └── val/
    │               └── <class_id>/
    │                   └── <filename>.xml
    └── LOC_synset_mapping.txt
    """

    # img path: root folder -> ILSVRC -> Data -> CLS-LOC -> test/train/val -> class_folders -> filename.JPEG
    # annotation path for obj detection: root folder -> ILSVRC -> Annotations -> CLS-LOC -> train/val -> class_folders -> filename.xml
    def __init__(
        self,
        root_dir: str,
        partition: str = "train",
        transforms=None,
        object_detection=False,
    ) -> None:
        self.img_dir = pathlib.Path(
            os.path.join(root_dir, "ILSVRC", "Data", "CLS-LOC", partition)
        )
        self.object_detection = object_detection
        if object_detection:
            self.annotation_dir = pathlib.Path(
                os.path.join(root_dir, "Annotations", "CLS-LOC", partition)
            )
            self.annotation_paths = [
                f for f in self.annotation_dir.rglob("*") if f.suffix.lower() == ".xml"
            ]
            self.img_paths = []

            for path in self.annotation_paths:
                path_parts = list(path.parts)
                data_index = path_parts.index("Annotations")
                path_parts[data_index] = "Data"
                img_base_path = pathlib.Path(*path_parts)
                self.img_paths.append(img_base_path.with_suffix(".JPEG"))
        else:
            self.img_paths = [f for f in self.img_dir.rglob("*") if f.suffix == ".JPEG"]

        self.readable_classes_dict = self.extract_readable_imagenet_labels(
            os.path.join(root_dir, "LOC_synset_mapping.txt")
        )
        self.transforms = transforms
        self.classes, self.class_to_idx = self.find_classes(
            self.img_dir, self.readable_classes_dict
        )

    def extract_readable_imagenet_labels(self, file_path: os.path) -> dict:
        """
        Helper function for storing imagenet human read-able
        class mappings. Mapping downloaded from
        https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
        """
        class_dict = {}

        with open(file_path, "r") as file:
            for line in file:
                words = line.strip().split()
                class_dict[words[0]] = words[1].rstrip(
                    ","
                )  # Incase there are several readable labels which are comma separated.

        return class_dict
    
    def find_classes(
        self, directory: str, readable_classes_dict: dict
    ) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        readable_classes = [readable_classes_dict.get(key) for key in classes]

        if not readable_classes:
            raise FileNotFoundError(f"Couldn't find any classes in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(readable_classes)}
        return readable_classes, class_to_idx

    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.img_paths[index]
        return Image.open(image_path).convert("RGB")

    def load_bounding_box_coords(self, index: int, img_size: Tuple) -> torch.Tensor:
        annotation_path = self.annotation_paths[index]
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        bndbox = root.find(".//bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        return torchvision.tv_tensors.BoundingBoxes(
            [[xmin, ymin, xmax, ymax]], format="XYXY", canvas_size=img_size
        )

    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.img_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name = self.img_paths[
            index
        ].parent.name  # expects path in data_folder/class_name/image.jpeg
        readable_class_name = self.readable_classes_dict[class_name]
        class_idx = self.class_to_idx[readable_class_name]

        if self.object_detection:
            H, W = img.height, img.width
            bndbox_coords_tensor = self.load_bounding_box_coords(index, (H, W))

            if self.transforms:
                return self.transforms(
                    {
                        "image": img,
                        "boxes": bndbox_coords_tensor,
                        "labels": torch.tensor([class_idx]),
                    }
                )
            else:
                return img, bndbox_coords_tensor, class_idx
        else:
            if self.transforms:
                return self.transforms(img), class_idx
            else:
                return img, class_idx