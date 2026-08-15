from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_images(folder: str | Path) -> list[Path]:
    root = Path(folder)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def prepare_image(path: Path, size: int, augment: bool = False) -> torch.Tensor:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        edge = min(width, height)
        if augment and min(width, height) > size:
            left = random.randint(0, width - edge)
            top = random.randint(0, height - edge)
        else:
            left, top = (width - edge) // 2, (height - edge) // 2
        image = image.crop((left, top, left + edge, top + edge))
        if augment and random.random() < 0.5:
            image = ImageOps.mirror(image)
        image = image.resize((size, size), Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.float32).copy() / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().float().cpu().clamp(-1, 1)
    array = ((tensor.permute(1, 2, 0) + 1) * 127.5).byte().numpy()
    return Image.fromarray(array, "RGB")


class ImageFolderDataset(Dataset):
    def __init__(self, folder: str | Path, size: int, augment: bool = True):
        self.paths = find_images(folder)
        if not self.paths:
            raise ValueError(f"No supported images were found in {folder}")
        self.size = size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        return prepare_image(self.paths[index], self.size, self.augment)


class TokenDataset(Dataset):
    def __init__(self, cache_path: str | Path):
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        self.tokens = payload["tokens"].long()
        self.meta = payload["meta"]

    def __len__(self) -> int:
        return self.tokens.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.tokens[index]
