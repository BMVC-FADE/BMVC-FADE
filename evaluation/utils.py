from bisect import bisect
from typing import Union, List

import numpy as np
import pandas as pd

from evaluation import metrics
from evaluation.metrics import compute_pro


def trapezoid(x, y, x_max=None):
    """
    This function calculates the definit integral of a curve given by
    x- and corresponding y-values. In contrast to, e.g., 'numpy.trapz()',
    this function allows to define an upper bound to the integration range by
    setting a value x_max.

    Points that do not have a finite x or y value will be ignored with a
    warning.

    Args:
        x: Samples from the domain of the function to integrate
          Need to be sorted in ascending order. May contain the same value
          multiple times. In that case, the order of the corresponding
          y values will affect the integration with the trapezoidal rule.
        y: Values of the function corresponding to x values.
        x_max: Upper limit of the integration. The y value at max_x will be
          determined by interpolating between its neighbors. Must not lie
          outside of the range of x.

    Returns:
        Area under the curve.
    """

    x = np.asarray(x)
    y = np.asarray(y)
    finite_mask = np.logical_and(np.isfinite(x), np.isfinite(y))
    if not finite_mask.all():
        print(
            "WARNING: Not all x and y values passed to trapezoid(...)"
            " are finite. Will continue with only the finite values."
        )
    x = x[finite_mask]
    y = y[finite_mask]

    # Introduce a correction term if max_x is not an element of x.
    correction = 0.0
    if x_max is not None:
        if x_max not in x:
            # Get the insertion index that would keep x sorted after
            # np.insert(x, ins, x_max).
            ins = bisect(x, x_max)
            # x_max must be between the minimum and the maximum, so the
            # insertion_point cannot be zero or len(x).
            assert 0 < ins < len(x)

            # Calculate the correction term which is the integral between
            # the last x[ins-1] and x_max. Since we do not know the exact value
            # of y at x_max, we interpolate between y[ins] and y[ins-1].
            y_interp = y[ins - 1] + (
                (y[ins] - y[ins - 1]) * (x_max - x[ins - 1]) / (x[ins] - x[ins - 1])
            )
            correction = 0.5 * (y_interp + y[ins - 1]) * (x_max - x[ins - 1])

        # Cut off at x_max.
        mask = x <= x_max
        x = x[mask]
        y = y[mask]

    # Return area under the curve using the trapezoidal rule.
    return np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction


def compute_and_store_final_results(results: List[dict]) -> pd.DataFrame:
    mean_metrics = {"object_name": "mean"}
    metric_names = list(results[-1].keys())[1:]
    for i, result_key in enumerate(metric_names):
        mean_metrics[result_key] = np.mean([x[result_key] for x in results])

    header = list(results[-1].keys())
    dt = pd.DataFrame(results + [mean_metrics], columns=header)
    return dt


def evaluation(
    ground_truth_labels: Union[list, np.array],
    predicted_labels: Union[list, np.array],
    ground_truth_segmentations: Union[List[np.array], np.array],
    predicted_segmentations: Union[List[np.array], np.array],
    integration_limit: float = 0.3,
) -> dict:
    full_image_auroc = np.nan
    full_image_aupr = np.nan
    full_image_f1_max = np.nan
    full_pixel_auroc = np.nan
    full_pixel_f1_max = np.nan
    full_pixel_au_pro = np.nan
    anomaly_pixel_auroc = np.nan
    anomaly_pixel_f1_max = np.nan

    if len(ground_truth_labels) != 0 and len(predicted_labels) != 0:
        # Compute image-level Auroc for all images
        image_scores = metrics.compute_imagewise_retrieval_metrics(
            predicted_labels, ground_truth_labels
        )
        full_image_auroc = image_scores["auroc"]
        full_image_aupr = image_scores["aupr"]
        full_image_f1_max = image_scores["f1_max"]

    if len(ground_truth_segmentations) != 0 and len(predicted_segmentations) != 0:
        # Compute PW Auroc for all images
        pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
            predicted_segmentations, ground_truth_segmentations
        )
        full_pixel_auroc = pixel_scores["auroc"]
        full_pixel_f1_max = pixel_scores["f1_max"]

        pro_curve = compute_pro(
            anomaly_maps=predicted_segmentations,
            ground_truth_maps=ground_truth_segmentations,
        )

        # Compute the area under the PRO curve.
        full_pixel_au_pro = trapezoid(
            pro_curve[0], pro_curve[1], x_max=integration_limit
        )
        full_pixel_au_pro /= integration_limit

        # Compute PRO score & PW Auroc only for images with anomalies
        sel_idxs = []
        for i in range(len(ground_truth_segmentations)):
            if np.sum(ground_truth_segmentations[i]) > 0:
                sel_idxs.append(i)
        pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
            [predicted_segmentations[i] for i in sel_idxs],
            [ground_truth_segmentations[i] for i in sel_idxs],
        )
        anomaly_pixel_auroc = pixel_scores["auroc"]
        anomaly_pixel_f1_max = pixel_scores["f1_max"]

    return {
        "full_image_auroc": full_image_auroc,
        "full_image_aupr": full_image_aupr,
        "full_image_f1_max": full_image_f1_max,
        "full_pixel_auroc": full_pixel_auroc,
        "full_pixel_f1_max": full_pixel_f1_max,
        "full_pixel_au_pro": full_pixel_au_pro,
        "anomaly_pixel_auroc": anomaly_pixel_auroc,
        "anomaly_pixel_f1_max": anomaly_pixel_f1_max,
    }
