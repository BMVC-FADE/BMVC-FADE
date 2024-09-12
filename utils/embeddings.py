import json
import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def extract_text_embeddings_by_batch(gem_model, text, batch_size=64):
    num_batches = math.ceil(len(text) / batch_size)
    text_embeddings = []
    for i in range(num_batches):
        text_embeddings.extend(
            gem_model.encode_text(text[i * batch_size : (i + 1) * batch_size])
            .squeeze(0)
            .tolist()
        )
    return text_embeddings


def extract_text_embeddings(prompt_path, gem_model, classname="", return_type="pandas"):
    with open(prompt_path) as fp:
        prompts = json.load(fp)

    with torch.no_grad():
        normal_text = [
            t.format(classname=classname) for t in prompts["normal"]["prompts"]
        ]
        abnormal_text = [
            t.format(classname=classname) for t in prompts["abnormal"]["prompts"]
        ]
        normal_text_embeddings = extract_text_embeddings_by_batch(
            gem_model, normal_text, batch_size=64
        )
        abnormal_text_embeddings = extract_text_embeddings_by_batch(
            gem_model, abnormal_text, batch_size=64
        )
        if return_type == "pandas":
            return pd.DataFrame(
                {
                    "text": normal_text + abnormal_text,
                    "feature": normal_text_embeddings + abnormal_text_embeddings,
                    "feature_type": "text",
                    "feature_label": ["good"] * len(normal_text_embeddings)
                    + ["anomaly"] * len(abnormal_text_embeddings),
                }
            )
        elif return_type == "numpy":
            X = np.array(normal_text_embeddings + abnormal_text_embeddings)
            y = np.array(
                [0] * len(normal_text_embeddings) + [1] * len(abnormal_text_embeddings)
            )
            return X, y


def extract_all_text_embeddings(prompt_paths, gem_model, classname=""):
    all_text_embeddings, all_text_labels = [], []
    for prompt_path in prompt_paths:
        text_embeddings, text_labels = extract_text_embeddings(
            prompt_path, gem_model, classname, return_type="numpy"
        )
        all_text_embeddings.append(text_embeddings)
        all_text_labels.append(text_labels)
    all_text_embeddings = np.concatenate(all_text_embeddings, axis=0)
    all_text_labels = np.concatenate(all_text_labels, axis=0)
    return all_text_embeddings, all_text_labels


def extract_image_embeddings(
    multiscale_images: dict[int, torch.Tensor], gem_model, device
) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    with torch.no_grad():
        features = {}
        for img_size, images in multiscale_images.items():
            features_gem, features_clip = gem_model.model.visual(images.to(device))
            features_gem = F.normalize(features_gem, dim=-1).detach().cpu().numpy()
            features_clip = F.normalize(features_clip, dim=-1).detach().cpu().numpy()
            features[img_size] = {
                "gem": {
                    "cls": features_gem[:, 0, :],
                    "patch": features_gem[:, 1:, :],
                },
                "clip": {
                    "cls": features_clip[:, 0, :],
                    "patch": features_clip[:, 1:, :],
                },
            }
    return features


def retrieve_image_embeddings(
    image_embeddings: dict[int, dict[str, dict[str, np.ndarray]]],
    img_size: int,
    feature_type: str,
    token_type: str,
) -> np.ndarray:
    """
    Args:
        image_embeddings: All image embeddings.
        img_size: integer indicating image size
        feature_type: 'clip' or 'gem'
        token_type: 'cls' or 'patch'

    Returns:
        Numpy array of dimension
        (batch_size, emb_dim) when token_type='cls'
        (batch_size, num_patches, emb_dim) when token_type='patch'
    """
    return image_embeddings[img_size][feature_type][token_type]
