from collections import defaultdict

import gem
import numpy as np
import torch

import datasets.base
from utils.embeddings import extract_image_embeddings, retrieve_image_embeddings


class ImageModel:
    """
    This is the image patch memory bank built from the reference images. predict_proba() computes the distance between
    a query patch and its kth nearest neighbour in the memory bank as the anomaly score.
    """

    def __init__(self, kth: int = 1):
        """
        Args:

            kth: predict_proba() will compute the distance to the kth nearest neighbour in the memory bank
        """
        self.kth = kth

    def fit(self, train_image_embeddings: np.ndarray):
        """
        Args:

            train_image_embeddings: Patch embeddings from training images.
                                    Numpy array of dimension (num_samples*num_patches, emb_dim)
        """
        self.train_image_embeddings = train_image_embeddings

    def predict_proba(self, image_embeddings: np.ndarray) -> np.ndarray:
        """
        Args:

            image_embeddings: Patch embeddings from query images.
                              Numpy array of dimension (num_samples*num_patches, emb_dim)

        Returns:
            Anomaly scores of dimension (num_samples*num_patches)
        """
        cosine_sim = image_embeddings @ self.train_image_embeddings.T
        anomaly_scores = 0.5 * (1 - cosine_sim)
        return np.partition(anomaly_scores, self.kth - 1, axis=1)[:, self.kth - 1]


def extract_ref_patch_embeddings(
    ref_dataset: datasets.base.BaseDataset,
    gem_model: gem.gem_wrapper.GEMWrapper,
    feature_type: str,
    shots: int,
    seed: int,
    device: str,
) -> dict[int, np.ndarray]:
    """
    Extract multi-resolution patch embeddings from the few-shot reference images

    Args:

        ref_dataset: Full reference dataset
        gem_model: GEM model to extract patch embeddings
        feature_type: 'clip' or 'gem'
        shots: Number of few-shot examples to be sampled from training dataset
        seed: Random seed used for random sampling few--shot examples
        device: gpu or cpu

    Returns:
        Dictionary of numpy arrays containing embeddings extracted at different resolution or img_size.
        Key of dict is the img_size and value is the patch embeddings with dimension (num_samples,
        num_patches, emb_dim)
    """
    ref_dataloader = torch.utils.data.DataLoader(
        ref_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True,
    )
    data = next(iter(ref_dataloader))
    img_sizes = list(data["image"])
    patch_embeddings = defaultdict(list)
    torch.manual_seed(seed)
    for i, data in enumerate(ref_dataloader):
        multiscale_images = {sz: data["image"][sz] for sz in img_sizes}
        image_embeddings = extract_image_embeddings(
            multiscale_images, gem_model, device
        )
        for img_size in img_sizes:
            patch_embeddings[img_size].append(
                retrieve_image_embeddings(
                    image_embeddings,
                    img_size=img_size,
                    feature_type=feature_type,
                    token_type="patch",
                )
            )
        if i + 1 == shots:
            break
    return {
        img_size: np.concatenate(patch_embeddings[img_size]) for img_size in img_sizes
    }


def extract_query_patch_embeddings(
    image_embeddings: dict[int, dict[str, dict[str, torch.Tensor]]],
    img_sizes: list,
    feature_type: str,
) -> dict[int, np.ndarray]:
    """
    Extract multi-resolution patch embeddings from the query image
    """
    return {
        img_size: retrieve_image_embeddings(
            image_embeddings,
            img_size=img_size,
            feature_type=feature_type,
            token_type="patch",
        )
        for img_size in img_sizes
    }


def combine_patch_embeddings(
    ref_patch_embeddings: dict[int, np.ndarray],
    query_patch_embeddings: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    """
    Combine the reference and query patch embeddings
    """
    assert (
        ref_patch_embeddings.keys() == query_patch_embeddings.keys()
    ), "Different image scales in ref_patch_embeddings and query_patch_embeddings"
    return {
        img_size: np.concatenate(
            [ref_patch_embeddings[img_size], query_patch_embeddings[img_size]]
        )
        for img_size in ref_patch_embeddings
    }


def build_image_models(
    train_patch_embeddings: dict[int, np.ndarray],
    use_query_img_in_vision_memory_bank: bool,
) -> dict[int, ImageModel]:
    """
    Build multi-resolution image models using patch embeddings from reference and/or query images

    Args:

        train_patch_embeddings: Dictionary of numpy arrays containing embeddings extracted at different resolution or
                                img_size. Key of dict is the img_size and value is the patch embeddings with dimension
                                (num_samples, num_patches, emb_dim)
        use_query_img_in_vision_memory_bank: Whether the query image patch embeddings is used to build the memory bank
                                             for vision-guided anomaly classification and segmentation.

    Returns:
        Dictionary of ImageModel built from patch embeddings at different resolution or img_size.
        Key of dict is img_size and value is ImageModel
    """
    image_models = {}
    for img_size, patch_embeddings_single_resolution in train_patch_embeddings.items():
        num_samples, num_patches, emb_dim = patch_embeddings_single_resolution.shape
        # If query image is used to construct the memory bank, we need to find the distance to the 2nd nearest neighbour
        # because the 1st nearest neighbour will be the query patch itself.
        image_models[img_size] = (
            ImageModel(kth=2)
            if use_query_img_in_vision_memory_bank
            else ImageModel(kth=1)
        )
        image_models[img_size].fit(
            patch_embeddings_single_resolution.reshape((-1, emb_dim))
        )
    return image_models
