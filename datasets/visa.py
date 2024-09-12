from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

from datasets.base import BaseDataset, DatasetSplit


class VisADataset(BaseDataset):
    CLASSNAMES = [
        "candle",
        "capsules",
        "cashew",
        "chewinggum",
        "fryum",
        "macaroni1",
        "macaroni2",
        "pcb1",
        "pcb2",
        "pcb3",
        "pcb4",
        "pipe_fryum",
    ]

    def get_image_data(self):
        imgpaths_per_class = dict()
        maskpaths_per_class = dict()

        file = Path(self.source) / "split_csv" / "1cls.csv"

        dt = pd.read_csv(file)

        for i, row in dt.iterrows():
            classname, set, label, image_path, mask_path = row

            if classname not in self.classnames_to_use:
                continue
            if classname not in imgpaths_per_class:
                imgpaths_per_class[classname] = defaultdict(list)
                maskpaths_per_class[classname] = defaultdict(list)

            if label == "normal":
                label = "good"
            else:
                label = "anomaly"
            img_src_path = self.source / image_path
            if not pd.isna(mask_path) and mask_path:
                msk_src_path = self.source / mask_path
            else:
                msk_src_path = None

            if (self.split == DatasetSplit.TEST and set == "test") or (
                self.split in [DatasetSplit.TRAIN, DatasetSplit.VAL] and set == "train"
            ):
                imgpaths_per_class[classname][label].append(img_src_path)
                maskpaths_per_class[classname][label].append(msk_src_path)

        if self.train_val_split < 1.0:
            for classname in imgpaths_per_class:
                for label in imgpaths_per_class[classname]:
                    n_images = len(imgpaths_per_class[classname][label])
                    train_val_split_idx = int(n_images * self.train_val_split)
                    if self.split == DatasetSplit.TRAIN:
                        imgpaths_per_class[classname][label] = imgpaths_per_class[
                            classname
                        ][label][:train_val_split_idx]
                        maskpaths_per_class[classname][label] = maskpaths_per_class[
                            classname
                        ][label][:train_val_split_idx]
                    elif self.split == DatasetSplit.VAL:
                        imgpaths_per_class[classname][label] = imgpaths_per_class[
                            classname
                        ][label][train_val_split_idx:]
                        maskpaths_per_class[classname][label] = maskpaths_per_class[
                            classname
                        ][label][train_val_split_idx:]

        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate
