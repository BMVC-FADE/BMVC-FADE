import os
from enum import Enum
from pathlib import Path
from typing import List, Union

import PIL
import torch
from torchvision import transforms
from torchvision.transforms.v2.functional import pil_to_tensor

# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD = [0.229, 0.224, 0.225]
# OpenCLIP preprocessing
IMAGENET_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGENET_STD = [0.26862954, 0.26130258, 0.27577711]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class BaseDataset(torch.utils.data.Dataset):
    """
    Base dataset for AD.
    """

    def __init__(
        self,
        source: Path,
        classname: Union[str, List],
        resize: Union[int, list] = 256,
        square: bool = True,
        split: DatasetSplit = DatasetSplit.TRAIN,
        train_val_split: float = 1.0,
        rotate_degrees: float = 0,
        translate: float = 0,
        brightness_factor: float = 0,
        contrast_factor: float = 0,
        saturation_factor: float = 0,
        gray_p: float = 0,
        h_flip_p: float = 0,
        v_flip_p: float = 0,
        scale: float = 0,
        **kwargs,
    ):
        """
        Args:
            source: [Path]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int or list[int]]. Size the loaded image initially gets resized to.
                    If square=True, image is resized to a square with side=resize
                    If square=False, smaller edge of the image will be matched to resize, maintaining aspect ratio
            square: [bool]. Whether to resize to a square or non-square image.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = (
            [classname] if not isinstance(classname, list) else classname
        )
        self.train_val_split = train_val_split
        self.resize = resize

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.random_transform_img = [
            # transforms.RandomRotation(
            #   rotate_degrees, transforms.InterpolationMode.BILINEAR
            # ),
            transforms.ColorJitter(
                brightness_factor, contrast_factor, saturation_factor
            ),
            transforms.RandomHorizontalFlip(h_flip_p),
            transforms.RandomVerticalFlip(v_flip_p),
            transforms.RandomGrayscale(gray_p),
            transforms.RandomAffine(
                rotate_degrees,
                translate=(translate, translate),
                scale=(1.0 - scale, 1.0 + scale),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
        ]
        self.random_transform_img = transforms.Compose(self.random_transform_img)

        if not isinstance(resize, list):
            resize = [resize]

        self.transform_img = []
        self.transform_mask = []

        # Multiple resize transforms
        for sz in resize:
            transform_img = [
                transforms.Resize(
                    (sz, sz) if square else sz,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                # transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
            self.transform_img.append(transforms.Compose(transform_img))

            transform_mask = [
                transforms.Resize((sz, sz) if square else sz),
                # transforms.CenterCrop(imagesize),
            ]
            self.transform_mask.append(transforms.Compose(transform_mask))

    def __getitem__(self, idx):
        resize = self.resize if isinstance(self.resize, list) else [self.resize]

        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]

        image = PIL.Image.open(image_path).convert("RGB")
        original_img_width, original_img_height = image.size
        image = self.random_transform_img(image)
        image = {
            sz: transform_img(image)
            for sz, transform_img in zip(resize, self.transform_img)
        }

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = (pil_to_tensor(mask) != 0).float()
        else:
            mask = torch.zeros([1, original_img_height, original_img_width])
        mask = {
            sz: (transform_mask(mask) > 0.5).float()
            for sz, transform_mask in zip(resize, self.transform_mask)
        }

        if not isinstance(self.resize, list):
            image = next(iter(image.values()))
            mask = next(iter(mask.values()))

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": os.path.relpath(image_path, self.source).split(".")[0],
            "image_path": str(image_path),
            "mask_path": "None" if mask_path is None else str(mask_path),
            "original_img_height": original_img_height,
            "original_img_width": original_img_width,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        raise NotImplementedError
