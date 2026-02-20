import pathlib
import re
from typing import Callable

import numpy as np
import torch
from PIL import Image


class TumorSegmentationDataset(torch.utils.data.Dataset):
    """Tumor segmentation dataset over controls and patients.

    Directory layout expected under data_root_dir_path:
    - controls/imgs/*.png
    - patients/imgs/*.png
    - patients/labels/*.png

    Controls have no provided labels, so they are assigned all-zero masks.
    """

    def __init__(
        self,
        data_root_dir_path: pathlib.Path,
        image_mask_transform: Callable[
            [torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
        ]
        | None = None,
    ):
        self.data_root_dir_path = pathlib.Path(data_root_dir_path)
        self.image_mask_transform = image_mask_transform

        self.controls_img_dir = self.data_root_dir_path / "controls" / "imgs"
        self.patient_img_dir = self.data_root_dir_path / "patients" / "imgs"
        self.patient_label_dir = self.data_root_dir_path / "patients" / "labels"

        for directory in [
            self.controls_img_dir,
            self.patient_img_dir,
            self.patient_label_dir,
        ]:
            if not directory.exists():
                raise FileNotFoundError(f"Missing directory: {directory}")

        self.samples = self._build_samples()
        if not self.samples:
            raise RuntimeError(
                f"No PNG files found under {self.data_root_dir_path}"
            )

    @staticmethod
    def _extract_index(stem: str) -> str:
        match = re.search(r"(\d+)", stem)
        if match is None:
            raise ValueError(f"Could not parse numeric id from: {stem}")
        return match.group(1)

    def _build_samples(self):
        samples = []

        control_image_paths = sorted(self.controls_img_dir.glob("*.png"))
        for image_path in control_image_paths:
            samples.append(
                {
                    "image_path": image_path,
                    "label_path": None,
                    "is_control": True,
                }
            )

        label_path_by_index = {
            self._extract_index(label_path.stem): label_path
            for label_path in self.patient_label_dir.glob("*.png")
        }

        patient_image_paths = sorted(self.patient_img_dir.glob("*.png"))
        for image_path in patient_image_paths:
            sample_index = self._extract_index(image_path.stem)
            label_path = label_path_by_index.get(sample_index)
            if label_path is None:
                raise FileNotFoundError(
                    f"Missing label for patient image: {image_path.name}"
                )
            samples.append(
                {
                    "image_path": image_path,
                    "label_path": label_path,
                    "is_control": False,
                }
            )

        return samples

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load_rgb_image(path: pathlib.Path) -> np.ndarray:
        image = Image.open(path).convert("RGB")
        return np.array(image, dtype=np.uint8, copy=True)

    @staticmethod
    def _load_binary_mask(path: pathlib.Path) -> np.ndarray:
        label_rgb = Image.open(path).convert("RGB")
        label_np = np.array(label_rgb, dtype=np.uint8, copy=True)
        # Any non-black pixel is treated as tumor.
        return (label_np.max(axis=2) > 0).astype(np.uint8)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        image_np = self._load_rgb_image(sample["image_path"])

        if sample["is_control"]:
            mask_np = np.zeros(image_np.shape[:2], dtype=np.uint8)
        else:
            mask_np = self._load_binary_mask(sample["label_path"])
            if mask_np.shape != image_np.shape[:2]:
                raise ValueError(
                    f"Mask/image shape mismatch for {sample['image_path'].name}: "
                    f"{mask_np.shape} vs {image_np.shape[:2]}"
                )

        image_np = np.ascontiguousarray(image_np)
        mask_np = np.ascontiguousarray(mask_np)

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()

        if self.image_mask_transform is not None:
            image_tensor, mask_tensor = self.image_mask_transform(
                image_tensor,
                mask_tensor,
            )

        return image_tensor, mask_tensor
