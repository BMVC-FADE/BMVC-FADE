import numpy as np
import torch
from open_clip import OPENAI_DATASET_STD, OPENAI_DATASET_MEAN

from datasets.mvtec import MVTecDataset
from datasets.visa import VisADataset


def undo_transform(image: torch.Tensor, unorm: bool = True) -> np.array:
    if unorm:
        image = (
            image * torch.Tensor(OPENAI_DATASET_STD)[:, None, None]
        ) + torch.Tensor(OPENAI_DATASET_MEAN)[:, None, None]
    return (image.permute(1, 2, 0) * 255).type(torch.uint8).numpy()


def min_max_normalization(arr: np.ndarray) -> np.ndarray:
    # Normalization per image in the batch
    # arr: (batch_size, height, width)
    arr_min = arr.min(axis=(1, 2), keepdims=True)
    arr_max = arr.max(axis=(1, 2), keepdims=True)
    return (arr - arr_min) / (arr_max - arr_min)
