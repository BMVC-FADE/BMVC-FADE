from collections import defaultdict
from pathlib import Path

import torch

from datasets.base import BaseDataset, DatasetSplit


class MVTecDataset(BaseDataset):
    CLASSNAMES = [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ]

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}
        catalog = defaultdict(list)

        for classname in self.classnames_to_use:
            classpath = self.source / classname / self.split.value
            maskpath = self.source / classname / "ground_truth"
            anomaly_paths = classpath.glob("*")

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly_path in anomaly_paths:
                anomaly = anomaly_path.name

                anomaly_files = sorted(anomaly_path.glob("*"))

                imgpaths_per_class[classname][anomaly] = anomaly_files

                catalog["classname"].append([classname] * len(anomaly_files))
                catalog["anomaly"].extend([anomaly] * len(anomaly_files))
                catalog["img_path"].extend(anomaly_files)

                if self.train_val_split < 1.0:
                    n_images = len(imgpaths_per_class[classname][anomaly])
                    train_val_split_idx = int(n_images * self.train_val_split)
                    if self.split == DatasetSplit.TRAIN:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][:train_val_split_idx]
                    elif self.split == DatasetSplit.VAL:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][train_val_split_idx:]

                if self.split == DatasetSplit.TEST and anomaly != "good":
                    anomaly_mask_path = maskpath / anomaly
                    anomaly_mask_files = sorted(anomaly_mask_path.glob("*"))

                    maskpaths_per_class[classname][anomaly] = anomaly_mask_files
                else:
                    maskpaths_per_class[classname]["good"] = None

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
