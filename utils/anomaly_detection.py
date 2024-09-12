from typing import Union

import cv2
import numpy as np

from utils.embeddings import retrieve_image_embeddings
from utils.image_model import ImageModel
from utils.text_model import TextModel


def predict_classification(
    text_model: TextModel,
    image_embeddings: dict,
    img_size: int,
    feature_type: str,
) -> np.ndarray:
    """
    Language-guided zero-shot classification
    Perform classification at a single image resolutions

    Args:
        text_model: Text embedding model
        image_embeddings: Dictionary of all embeddings
        img_size: Image resolution of the embeddings to use
        feature_type: 'clip' or 'gem'

    Returns:
        Classification scores of dimension (batch_size)
    """
    cls_embeddings = retrieve_image_embeddings(
        image_embeddings,
        img_size=img_size,
        feature_type=feature_type,
        token_type="cls",
    )
    scores = text_model.predict_proba(cls_embeddings)
    return scores[:, 1]


def text_image_matching(
    text_model: TextModel,
    patch_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Segmentation by matching image patch embeddings to the text embeddings.

    Args:
        text_model: Text embedding model
        patch_embeddings: Query patch embeddings of dimensions (batch_size, num_patches, emb_dim)

    Returns:
        Segmentation scores of dimension (batch_size, num_patches)
    """
    batch_size, num_patches, emb_dim = patch_embeddings.shape
    segmentations = text_model.predict_proba(
        patch_embeddings.reshape((batch_size * num_patches, emb_dim))
    ).reshape((batch_size, num_patches, 2))
    return segmentations[..., 1]


def image_image_matching(
    image_model: ImageModel,
    patch_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Segmentation by matching image patch embeddings to the training image patch embeddings.

    Args:
        image_model: Image embedding model
        patch_embeddings: Query patch embeddings of dimensions (batch_size, num_patches, emb_dim)

    Returns:
        Segmentation scores of dimension (batch_size, num_patches)
    """
    batch_size, num_patches, emb_dim = patch_embeddings.shape
    segmentations = image_model.predict_proba(
        patch_embeddings.reshape((batch_size * num_patches, emb_dim))
    ).reshape((batch_size, num_patches))
    return segmentations


def resize_and_aggregate_segmentations(
    segmentations_multi_resolution: list[np.ndarray],
    output_size: int,
) -> np.ndarray:
    """
    Resize and aggregate segmentations of different resolutions by batch.

    Args:
        segmentations_multi_resolution: A list of arrays where each array contains segmentations of
                                        one resolution. Each array has dimension (batch_size, height, width).
        output_size: The size to resize the segmentations to before aggregation.

    Returns:
        Segmentations array of dimension (batch_size, output_size, output_size)
    """
    batch_size = segmentations_multi_resolution[0].shape[0]
    return np.stack(
        [
            np.mean(
                [
                    cv2.resize(
                        segmentations_single_resolution[i], (output_size, output_size)
                    )
                    for segmentations_single_resolution in segmentations_multi_resolution
                ],
                axis=0,
            )
            for i in range(batch_size)
        ]
    )


def predict_segmentation(
    model: Union[TextModel, dict[int, ImageModel]],
    image_embeddings: dict,
    img_sizes: list[int],
    feature_type: str,
    patch_size: tuple[int, int],
    segmentation_mode: str,
) -> np.ndarray:
    """
    Language-guided (zero-shot) or vision-guided (few-shot) segmentation
    Perform segmentation at multiple image resolutions. Then, resize and aggregate into a single segmentation map.

    Args:
        model: Model used for the segmentation
               A single text embedding model if segmentation_mode='language'.
               A dictionary of multi-resolution image embedding models if segmentation_mode='vision'.
        image_embeddings: Dictionary of all embeddings
        img_sizes: List of image resolutions of the embeddings to use
        feature_type: 'clip' or 'gem'
        patch_size: Patch size used in the GEM/CLIP model
        segmentation_mode: 'language': Zero-shot language-guided segmentation
                           'vision': Few-shot vision-guided segmentation

    Returns:
        Segmentation scores of dimension (batch_size, height, width)
        height and width correspond to the highest resolution segmentation
    """
    assert (segmentation_mode == "language" and isinstance(model, TextModel)) or (
        segmentation_mode == "vision"
        and isinstance(model, dict)
        and all(isinstance(x, ImageModel) for x in model.values())
    )

    # Anomaly segmentation at multiple resolutions
    segmentations_multi_resolution = []
    for img_size in img_sizes:
        patch_embeddings = retrieve_image_embeddings(
            image_embeddings,
            img_size=img_size,
            feature_type=feature_type,
            token_type="patch",
        )
        if segmentation_mode == "language":
            segmentations_flatten = text_image_matching(model, patch_embeddings)
        elif segmentation_mode == "vision":
            segmentations_flatten = image_image_matching(
                model[img_size], patch_embeddings
            )
        else:
            raise ValueError(
                f"segmentation_mode can only be set to 'language' or 'vision'. ({segmentation_mode=})"
            )
        segmentations_single_resolution = segmentations_flatten.reshape(
            (
                -1,
                img_size // patch_size[0],
                img_size // patch_size[1],
            )
        )
        segmentations_multi_resolution.append(segmentations_single_resolution)

    # Resize and aggregate segmentation of multiple resolutions
    return resize_and_aggregate_segmentations(
        segmentations_multi_resolution,
        max(img_sizes) // patch_size[0],  # Resize to largest resolution
    )
