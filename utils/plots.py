from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_plot(fig, save_path):
    save_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(save_path, format="png")


def plot_segmentation_images(
    image_destination: Path,
    image_names,
    images,
    segmentations,
    anomaly_scores=None,
    masks=None,
    image_transform=lambda x: x,
    mask_transform=lambda x: x,
):
    """Generate anomaly segmentation images.

    Args:
        image_names: List[str] List of image names.
        images: List[np.ndarray] List of images.
        segmentations: [List[np.ndarray]] Generated anomaly segmentations.
        anomaly_scores: [List[float]] Anomaly scores for each image.
        masks: [List[np.ndarray]] List of ground truth masks.
        image_transform: [function or lambda] Optional transformation of images.
        mask_transform: [function or lambda] Optional transformation of masks.
    """
    vis_paths = []
    if anomaly_scores is None:
        anomaly_scores = [np.nan for _ in range(len(image_names))]
    for image_name, image, mask, anomaly_score, segmentation in zip(
        image_names, images, masks, anomaly_scores, segmentations
    ):
        image = image_transform(image)
        mask = mask_transform(mask)
        heatmap = cv2.cvtColor(
            cv2.applyColorMap(segmentation, cv2.COLORMAP_JET),
            cv2.COLOR_BGR2RGB,
        )
        superimposed = cv2.addWeighted(heatmap, 0.5, image, 0.5, 0)

        f, axes = plt.subplots(1, 4)
        axes[0].imshow(image)
        axes[0].set_title(image_name)
        axes[0].axis("off")
        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("GT")
        axes[1].axis("off")
        axes[2].imshow(segmentation, cmap="gray", vmin=0, vmax=255)
        axes[2].set_title(f"Prediction\nImage-level score = {anomaly_score:.4f}")
        axes[2].axis("off")
        axes[3].imshow(superimposed)
        axes[3].set_title("Prediction overlaid")
        axes[3].axis("off")
        f.set_size_inches(3 * 4, 3)
        f.tight_layout()
        save_plot(f, image_destination / (image_name + ".png"))
        plt.close()
        vis_paths.append(image_destination / (image_name + ".png"))
    return vis_paths
