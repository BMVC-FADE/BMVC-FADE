from collections import defaultdict
from pathlib import Path

import click
import cv2
import gem
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from pkg_resources import require
from tqdm import tqdm

from datasets.base import DatasetSplit, BaseDataset
from datasets.mvtec import MVTecDataset
from datasets.utils import undo_transform, min_max_normalization
from datasets.visa import VisADataset
from evaluation.utils import compute_and_store_final_results, evaluation
from utils.anomaly_detection import predict_classification, predict_segmentation
from utils.embeddings import extract_image_embeddings
from utils.image_model import (
    extract_ref_patch_embeddings,
    build_image_models,
    extract_query_patch_embeddings,
    combine_patch_embeddings,
)
from utils.plots import plot_segmentation_images
from utils.text_model import build_text_model


def load_dataset(dataset_name: str, dataset_source: str, **kwargs) -> BaseDataset:
    if dataset_name == "mvtec":
        return MVTecDataset(
            source=Path(dataset_source),
            **kwargs,
        )
    elif dataset_name == "visa":
        return VisADataset(
            source=Path(dataset_source),
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid {dataset_name=}")


def load_classnames(dataset_name: str) -> list:
    if dataset_name == "mvtec":
        return MVTecDataset.CLASSNAMES
    elif dataset_name == "visa":
        return VisADataset.CLASSNAMES
    else:
        raise ValueError(f"Invalid {dataset_name=}")


@click.command()
@click.option("--dataset-name", type=click.Choice(["mvtec", "visa"]), default="mvtec")
@click.option("--dataset-source", type=str, required=True)
@click.option(
    "--experiment-name",
    type=str,
    default="mvtec/fewshot/cm_both_sm_both/img_size_448/1shot/seed0",
)
@click.option("--model-name", type=str, default="ViT-B/16-plus-240")
@click.option(
    "--pretrained",
    type=str,
    default="models/openclip/clip/vit_b_16_plus_240-laion400m_e31-8fb26589.pt",
)
@click.option(
    "--classification-mode",
    type=click.Choice(["none", "language", "vision", "both"]),
    default="both",
)
@click.option(
    "--segmentation-mode",
    type=click.Choice(["none", "language", "vision", "both"]),
    default="both",
)
@click.option(
    "--language-classification-feature",
    type=click.Choice(["clip", "gem"]),
    default="clip",
    help="Feature used for language-guided anomaly classification",
)
@click.option(
    "--language-segmentation-feature",
    type=click.Choice(["clip", "gem"]),
    default="gem",
    help="Feature used for language-guided anomaly segmentation",
)
@click.option(
    "--vision-feature",
    type=click.Choice(["clip", "gem"]),
    default="gem",
    help="Feature used for vision-guided anomaly classification and segmentation"
    "Note that vision-guided classification and segmentation use the same feature",
)
@click.option(
    "--vision-segmentation-multiplier",
    type=float,
    default=3.5,
    help="A number multiplied to the vision-guided segmentation map to calibrate its upper bound value to be around 1",
)
@click.option(
    "--vision-segmentation-weight",
    type=click.FloatRange(0.0, 1.0),
    default=0.85,
    help="Weighting w given to the vision-guided segmentation map"
    "Only used when segmentation_mode='both'"
    "Segmentations are merged by: (1-w) * language_segmentation + w * vision_segmentation",
)
@click.option(
    "--use-query-img-in-vision-memory-bank/--no-use-query-img-in-vision-memory-bank",
    type=bool,
    default=False,
    help="Whether to use the query image patch embeddings to build the memory bank for vision-guided anomaly "
    "classification and segmentation."
    "Only used when classification_mode or segmentation_mode is set to 'vision' or 'both'",
)
@click.option(
    "--classification-img-size",
    type=int,
    default=240,
    help="Input image size of classification model",
)
@click.option(
    "--segmentation-img-sizes",
    type=str,
    default="240,448,896",
    help="Input image sizes of segmentation models",
)
@click.option(
    "--eval-img-size",
    type=int,
    default=448,
    help="Image size used for evaluation and visualisation",
)
@click.option("--square/--no-square", type=bool, default=True)
@click.option(
    "--text-model-type",
    type=click.Choice(
        ["average", "softmax", "max", "lr", "mlp", "knn", "rf", "xgboost", "gmm"]
    ),
    default="average",
)
@click.option(
    "--shots",
    type=int,
    default=1,
    help="Number of reference images for few-shot detection. "
    "Only used when classification_mode or segmentation_mode is set to 'vision' or 'both'",
)
@click.option(
    "--seed",
    type=int,
    default=0,
    help="Random seed for sampling the few-shot reference images",
)
@click.option(
    "--normalize-segmentations/--no-normalize-segmentations", type=bool, default=False
)
@click.option("--save-visualization/--no-save-visualization", type=bool, default=False)
@click.option("--save-segmentation/--no-save-segmentation", type=bool, default=False)
def main(
    dataset_name: str,
    dataset_source: str,
    experiment_name: str,
    model_name: str,
    pretrained: str,
    classification_mode: str,
    segmentation_mode: str,
    language_classification_feature: str,
    language_segmentation_feature: str,
    vision_feature: str,
    vision_segmentation_multiplier: float,
    vision_segmentation_weight: float,
    use_query_img_in_vision_memory_bank: bool,
    classification_img_size: int,
    segmentation_img_sizes: str,
    eval_img_size: int,
    square: bool,
    text_model_type: str,
    shots: int,
    seed: int,
    normalize_segmentations: bool,
    save_visualization: bool,
    save_segmentation: bool,
):
    print("Start processing...")
    result_destination = Path(experiment_name) / "result"
    result_destination.mkdir(parents=True, exist_ok=True)

    if save_visualization:
        image_destination = Path(experiment_name) / "images"
        image_destination.mkdir(parents=True, exist_ok=True)

    if save_segmentation:
        seg_destination = Path(experiment_name) / "segmentations"
        seg_destination.mkdir(parents=True, exist_ok=True)

    text_prompt_path = "prompts"
    # Text prompts used for anomaly classification
    prompt_paths_classification = [
        f"{text_prompt_path}/winclip_prompt.json",
    ]
    use_classname_in_prompt_classification = True

    # Text prompts used for anomaly segmentation
    prompt_paths_segmentation = [
        f"{text_prompt_path}/winclip_prompt.json",
        # f"{text_prompt_path}/manual_prompt.json",
        # f"{text_prompt_path}/manual_prompt_with_classname.json",
        # f"{text_prompt_path}/manual_prompt_with_classname2.json",
        # f"{text_prompt_path}/winclip_prompt_aug_size_position.json",
        f"{text_prompt_path}/chatgpt3.5_prompt1.json",
        f"{text_prompt_path}/chatgpt3.5_prompt2.json",
        f"{text_prompt_path}/chatgpt3.5_prompt3.json",
        f"{text_prompt_path}/chatgpt3.5_prompt4.json",
        f"{text_prompt_path}/chatgpt3.5_prompt5.json",
    ]
    use_classname_in_prompt_segmentation = False

    model_cache_dir = "models"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segmentation_img_sizes = [int(i) for i in segmentation_img_sizes.split(",")]

    config = {
        "dataset_name": dataset_name,
        "dataset_root": dataset_source,
        "experiment_name": experiment_name,
        "model_name": model_name,
        "pretrained": pretrained,
        "model_cache_dir": model_cache_dir,
        "classification_mode": classification_mode,
        "segmentation_mode": segmentation_mode,
        "language_classification_feature": language_classification_feature,
        "language_segmentation_feature": language_segmentation_feature,
        "vision_feature": vision_feature,
        "vision_segmentation_multiplier": vision_segmentation_multiplier,
        "vision_segmentation_weight": vision_segmentation_weight,
        "use_query_img_in_vision_memory_bank": use_query_img_in_vision_memory_bank,
        "classification_img_size": classification_img_size,
        "segmentation_img_sizes": segmentation_img_sizes,
        "eval_img_size": eval_img_size,
        "square": square,
        "prompt_paths_classification": prompt_paths_classification,
        "use_classname_in_prompt_classification": use_classname_in_prompt_classification,
        "prompt_paths_segmentation": prompt_paths_segmentation,
        "use_classname_in_prompt_segmentation": use_classname_in_prompt_segmentation,
        "text_model_type": text_model_type,
        "shots": shots,
        "seed": seed,
        "normalize_segmentations": normalize_segmentations,
        "save_visualization": save_visualization,
        "save_segmentation": save_segmentation,
        "device": device,
    }
    with open(result_destination / "config.yaml", "w") as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)

    # init model
    gem_model = gem.create_gem_model(
        model_name=model_name,
        pretrained=pretrained,
        device=device,
    )

    result_collect = []
    plot_data = defaultdict(list)

    classnames = load_classnames(dataset_name)
    for classname in classnames:
        print(f"Processing '{classname}'")
        # load image dataset
        dataset = load_dataset(
            dataset_name=dataset_name,
            dataset_source=dataset_source,
            classname=classname,
            resize=list(
                {classification_img_size, *segmentation_img_sizes, eval_img_size}
            ),
            square=square,
            split=DatasetSplit.TEST,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            prefetch_factor=2,
            pin_memory=True,
        )

        # Build text model using prompts for language-guided anomaly detection
        if classification_mode == "language" or classification_mode == "both":
            classification_text_model = build_text_model(
                gem_model=gem_model,
                prompt_paths=prompt_paths_classification,
                classname=(
                    classname.replace("_", " ")
                    if use_classname_in_prompt_classification
                    else "object"
                ),
                text_model_type=text_model_type,
            )
        if segmentation_mode == "language" or segmentation_mode == "both":
            segmentation_text_model = build_text_model(
                gem_model=gem_model,
                prompt_paths=prompt_paths_segmentation,
                classname=(
                    classname.replace("_", " ")
                    if use_classname_in_prompt_segmentation
                    else "object"
                ),
                text_model_type=text_model_type,
            )

        # Build image model using reference images for vision-guided anomaly detection
        ref_patch_embeddings = None
        if (
            classification_mode == "vision"
            or classification_mode == "both"
            or segmentation_mode == "vision"
            or segmentation_mode == "both"
        ) and shots > 0:
            ref_dataset = load_dataset(
                dataset_name=dataset_name,
                dataset_source=dataset_source,
                classname=classname,
                resize=segmentation_img_sizes,
                square=square,
                split=DatasetSplit.TRAIN,
            )
            ref_patch_embeddings = extract_ref_patch_embeddings(
                ref_dataset,
                gem_model,
                vision_feature,
                shots,
                seed,
                device,
            )

        anomaly_scores = []
        ground_truth_scores = []
        anomaly_segmentations = []
        ground_truth_segmentations = []

        for data in tqdm(dataloader):
            for key in [
                "classname",
                "anomaly",
                "is_anomaly",
                "image_name",
                "image_path",
                "mask_path",
            ]:
                plot_data[key].append(
                    data[key].tolist()
                    if isinstance(data[key], torch.Tensor)
                    else data[key]
                )

            # Extract image embeddings
            img_sizes = list({classification_img_size, *segmentation_img_sizes})
            multiscale_images = {sz: data["image"][sz] for sz in img_sizes}
            image_embeddings = extract_image_embeddings(
                multiscale_images, gem_model, device
            )

            # Language-guided anomaly classification
            language_guided_scores = None
            if classification_mode == "language" or classification_mode == "both":
                language_guided_scores = predict_classification(
                    text_model=classification_text_model,
                    image_embeddings=image_embeddings,
                    img_size=classification_img_size,
                    feature_type=language_classification_feature,
                )

            # Language-guided anomaly segmentation
            language_guided_maps = None
            if segmentation_mode == "language" or segmentation_mode == "both":
                language_guided_maps = predict_segmentation(
                    model=segmentation_text_model,
                    image_embeddings=image_embeddings,
                    img_sizes=segmentation_img_sizes,
                    feature_type=language_segmentation_feature,
                    patch_size=gem_model.model.visual.patch_size,
                    segmentation_mode="language",
                )

            # Vision-guided anomaly segmentation and classification
            vision_guided_scores = None
            vision_guided_maps = None
            if (
                classification_mode == "vision"
                or classification_mode == "both"
                or segmentation_mode == "vision"
                or segmentation_mode == "both"
            ):
                # Build image models using patch embeddings from reference and/or query images
                query_patch_embeddings = None
                if use_query_img_in_vision_memory_bank:
                    query_patch_embeddings = extract_query_patch_embeddings(
                        image_embeddings, segmentation_img_sizes, vision_feature
                    )
                if ref_patch_embeddings and query_patch_embeddings:
                    train_patch_embeddings = combine_patch_embeddings(
                        ref_patch_embeddings, query_patch_embeddings
                    )
                else:
                    train_patch_embeddings = (
                        ref_patch_embeddings or query_patch_embeddings
                    )
                assert (
                    train_patch_embeddings
                ), "You cannot set shots=0 AND use_query_img_in_vision_memory_bank=False"
                image_models = build_image_models(
                    train_patch_embeddings, use_query_img_in_vision_memory_bank
                )

                # Vision-guided anomaly segmentation
                vision_guided_maps = predict_segmentation(
                    model=image_models,
                    image_embeddings=image_embeddings,
                    img_sizes=segmentation_img_sizes,
                    feature_type=vision_feature,
                    patch_size=gem_model.model.visual.patch_size,
                    segmentation_mode="vision",
                )
                vision_guided_maps *= vision_segmentation_multiplier

                # Vision-guided anomaly classification
                if classification_mode == "vision" or classification_mode == "both":
                    vision_guided_scores = np.max(vision_guided_maps, axis=(1, 2))

            # Final classification scores
            scores = None
            if classification_mode != "none":
                if classification_mode == "language":
                    scores = language_guided_scores
                elif classification_mode == "vision":
                    scores = vision_guided_scores
                elif classification_mode == "both":
                    scores = (language_guided_scores + vision_guided_scores) / 2
                scores = np.clip(scores, 0, 1)
                plot_data["image_anomaly_score"].append(scores.tolist())
                anomaly_scores.append(scores)
                ground_truth_scores.append(data["is_anomaly"])

            # Final segmentation maps
            segmentations = None
            if segmentation_mode != "none":
                if segmentation_mode == "language":
                    segmentations = language_guided_maps
                elif segmentation_mode == "vision":
                    segmentations = vision_guided_maps
                elif segmentation_mode == "both":
                    segmentations = (
                        (1.0 - vision_segmentation_weight) * language_guided_maps
                        + vision_segmentation_weight * vision_guided_maps
                    )

                # Post-processing segmentation maps
                if normalize_segmentations:
                    segmentations = min_max_normalization(segmentations)
                segmentations = np.clip(segmentations, 0, 1)
                segmentations = (segmentations * 255).astype("uint8")

                if save_segmentation:
                    for seg, img_name in zip(segmentations, data["image_name"]):
                        img = Image.fromarray(seg).convert("RGB")
                        save_path = seg_destination / (img_name + ".png")
                        save_path.parent.mkdir(exist_ok=True, parents=True)
                        img.save(save_path)

                # Resize segmentation for evaluation and visualisation
                segmentations = np.array(
                    [
                        cv2.resize(seg, (eval_img_size, eval_img_size))
                        for seg in segmentations
                    ]
                )

                if save_visualization:
                    plot_data["vis_path"].append(
                        plot_segmentation_images(
                            image_destination=image_destination,
                            image_names=data["image_name"],
                            images=data["image"][eval_img_size],
                            segmentations=segmentations,
                            anomaly_scores=scores,
                            masks=data["mask"][eval_img_size],
                            image_transform=lambda x: undo_transform(x, unorm=True),
                            mask_transform=lambda x: undo_transform(x, unorm=False),
                        )
                    )

                anomaly_segmentations.append(segmentations)
                ground_truth_segmentations.append(
                    data["mask"][eval_img_size][:, 0, :, :]
                )

        # Evaluations
        if classification_mode != "none":
            anomaly_scores = np.concatenate(anomaly_scores)
            ground_truth_scores = np.concatenate(ground_truth_scores)
        if segmentation_mode != "none":
            anomaly_segmentations = np.concatenate(anomaly_segmentations)
            ground_truth_segmentations = np.concatenate(ground_truth_segmentations)
        object_results = evaluation(
            ground_truth_scores,
            anomaly_scores,
            ground_truth_segmentations,
            anomaly_segmentations,
        )
        result_collect.append({"object_name": classname, **object_results})
        print(f"Object: {classname}")
        print(f"Full image AUROC: {object_results['full_image_auroc']:.2f}")
        print(f"Full pixel AUROC: {object_results['full_pixel_auroc']:.2f}")
        print(f"Anomaly pixel AUROC: {object_results['anomaly_pixel_auroc']:.2f}")
        print("\n")

    results_dt = compute_and_store_final_results(result_collect)
    results_dt.to_csv(result_destination / "evaluation_results.csv", index=False)

    # Save results
    plot_data = {key: np.concatenate(plot_data[key]).tolist() for key in plot_data}
    plot_data = pd.DataFrame(plot_data)
    if "vis_path" not in plot_data:
        plot_data["vis_path"] = None

    plot_data.to_csv(result_destination / "plot_results.csv", index=False)


if __name__ == "__main__":
    main()
