from pathlib import Path
from typing import List, Union

import gem
import numpy as np
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier

from utils.embeddings import extract_all_text_embeddings


class TextModel:
    pass


class AverageTextModel(TextModel):
    """
    Compute a mean text embedding for each normal and abnormal group separately.
    Then compute cosine similarity between image embedding and each mean text embedding.
    Do a softmax to get the probs.
    """

    def fit(self, text_embeddings: np.ndarray, text_labels: np.ndarray):
        self.mean_text_embbeddings = np.stack(
            [
                np.mean(text_embeddings[text_labels == 0], axis=0),
                np.mean(text_embeddings[text_labels == 1], axis=0),
            ]
        )

    def predict_proba(self, image_embeddings: np.ndarray) -> np.ndarray:
        logits = 100.0 * image_embeddings @ self.mean_text_embbeddings.T
        return softmax(logits, axis=1)


class SoftmaxTextModel(TextModel):
    """
    Compute cosine similarity between image embedding and each prompt in the prompt ensemble.
    Then do a softmax across all prompts.
    Add up probs for normal and abnormal group separately
    """

    def fit(self, text_embeddings: np.ndarray, text_labels: np.ndarray):
        self.text_embeddings = text_embeddings
        self.text_labels = text_labels.astype(bool)

    def predict_proba(self, image_embeddings: np.ndarray) -> np.ndarray:
        logits = 100.0 * image_embeddings @ self.text_embeddings.T
        probs = softmax(logits, axis=1)
        normal_probs = probs[:, ~self.text_labels].sum(axis=1)
        abnormal_probs = probs[:, self.text_labels].sum(axis=1)
        return np.stack([normal_probs, abnormal_probs], axis=-1)


class MaxTextModel(TextModel):
    """
    For each image embedding, find its nearest text embedding for each normal and abnormal group by max cosine similarity.
    Use the cosine similarity wrt the nearest text embedding for each normal and abnormal group.
    Do a softmax to get the probs
    """

    def fit(self, text_embeddings: np.ndarray, text_labels: np.ndarray):
        self.text_embeddings = text_embeddings
        self.text_labels = text_labels.astype(bool)

    def predict_proba(self, image_embeddings: np.ndarray) -> np.ndarray:
        logits = 100.0 * image_embeddings @ self.text_embeddings.T
        normal_max = logits[:, ~self.text_labels].max(axis=1)
        abnormal_max = logits[:, self.text_labels].max(axis=1)
        return softmax(
            np.stack([normal_max, abnormal_max], axis=-1),
            axis=-1,
        )


class SupervisedModel(TextModel):
    """
    Train a binary supervised model on the text embeddings. Use the trained model to do inference on image embeddings.
    """

    def __init__(self, model_type):
        if model_type == "lr":
            self.model = LogisticRegression(
                solver="liblinear", penalty="l2", n_jobs=-1, random_state=42
            )
        elif model_type == "mlp":
            self.model = MLPClassifier(
                hidden_layer_sizes=640,
                max_iter=200,
                early_stopping=True,
                random_state=42,
                verbose=True,
            )
        elif model_type == "knn":
            self.model = KNeighborsClassifier(
                n_neighbors=5, weights="distance", metric="cosine", n_jobs=-1
            )
        elif model_type == "rf":
            self.model = RandomForestClassifier(n_jobs=-1, random_state=42)
        elif model_type == "xgboost":
            self.model = XGBClassifier(
                objective="binary:logistic", n_jobs=-1, random_state=42
            )
        else:
            raise f"Unknown {model_type=}"

    def fit(self, text_embeddings: np.ndarray, text_labels: np.ndarray):
        self.model.fit(text_embeddings, text_labels)

    def predict_proba(self, image_embeddings: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(image_embeddings)


class GmmModel(TextModel):
    def __init__(
        self, gmm_components, covariance_type, run_pca=True, pca_components=0.9
    ):
        normal_pca = (
            PCA(n_components=pca_components, random_state=42)
            if run_pca
            else FunctionTransformer()
        )
        abnormal_pca = (
            PCA(n_components=pca_components, random_state=42)
            if run_pca
            else FunctionTransformer()
        )
        normal_gmm = GaussianMixture(
            n_components=gmm_components,
            covariance_type=covariance_type,
            random_state=42,
        )
        abnormal_gmm = GaussianMixture(
            n_components=gmm_components,
            covariance_type=covariance_type,
            random_state=42,
        )
        self.normal_pipe = Pipeline([("pca", normal_pca), ("gmm", normal_gmm)])
        self.abnormal_pipe = Pipeline([("pca", abnormal_pca), ("gmm", abnormal_gmm)])

    def fit(self, text_embeddings: np.ndarray, text_labels: np.ndarray):
        mask = np.array(text_labels, dtype=bool)
        self.normal_pipe.fit(text_embeddings[np.logical_not(mask)])
        self.abnormal_pipe.fit(text_embeddings[mask])

    def predict_proba(self, image_embeddings: np.ndarray) -> np.ndarray:
        normal_logprobs = self.normal_pipe.score_samples(image_embeddings)
        abnormal_logprobs = self.abnormal_pipe.score_samples(image_embeddings)
        logprobs = np.stack([normal_logprobs, abnormal_logprobs], axis=-1)
        return softmax(logprobs, axis=-1)


def get_text_model(model_type):
    if model_type == "average":
        return AverageTextModel()
    elif model_type == "softmax":
        return SoftmaxTextModel()
    elif model_type == "max":
        return MaxTextModel()
    elif model_type in ["lr", "mlp", "knn", "rf", "xgboost"]:
        return SupervisedModel(model_type)
    elif model_type == "gmm":
        return GmmModel(
            gmm_components=10,
            covariance_type="spherical",
            run_pca=True,
            pca_components=0.9,
        )
    else:
        raise f"Unknown {model_type=}"


def build_text_model(
    gem_model: gem.gem_wrapper.GEMWrapper,
    prompt_paths: List[Union[Path, str]],
    classname: str,
    text_model_type: str,
) -> TextModel:
    # Text embeddings
    text_embeddings, text_labels = extract_all_text_embeddings(
        prompt_paths,
        gem_model,
        classname=classname,
    )
    # Text model
    text_model = get_text_model(model_type=text_model_type)
    text_model.fit(text_embeddings, text_labels)
    return text_model
