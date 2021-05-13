from __future__ import annotations
from collections import Counter

from lapjv import lapjv
import numba
import numba
import numpy as np
import numpy as np
from scipy.optimize import linear_sum_assignment

from stratification.cluster.fast_sil import silhouette_samples


__all__ = [
    "apply_encodiing_dict",
    "compute_optimal_assignments",
    "get_cluster_composition",
    "get_cluster_mean_loss",
    "get_k_from_model",
]


def get_k_from_model(model):
    if hasattr(model, 'n_clusters'):
        return model.n_clusters
    elif hasattr(model, 'n_components'):
        return model.n_components
    else:
        raise NotImplementedError(
            f'model {type(model)} K not found.'
            + f'model attributes:\n{list(model.__dict__.keys())}'
        )


def get_cluster_mean_loss(sample_losses, assignments):
    cluster_losses = {}

    C = np.unique(assignments)
    for c in C:
        cluster_loss = np.mean(sample_losses[assignments == c])
        cluster_losses[str(c)] = float(cluster_loss)
    return cluster_losses


def get_cluster_composition(superclasses, assignments):
    compositions = {}

    S = np.unique(superclasses)
    C = np.unique(assignments)
    for c in C:
        superclasses_c = superclasses[assignments == c]
        counts = dict(Counter(superclasses_c))
        compositions[str(c)] = {str(s): counts.get(s, 0) for s in S}
    return compositions


def compute_optimal_assignments(
    labels_pred: np.ndarray,
    labels_true: np.ndarray,
    num_classes: int | None = None,
    encode: bool = True,
) -> tuple[float, dict[int, int]]:
    """Find an assignment of cluster to class such that the overall accuracy is maximized."""
    # row_ind maps from class ID to cluster ID: cluster_id = row_ind[class_id]
    # col_ind maps from cluster ID to class ID: class_id = row_ind[cluster_id]
    cost_matrix, decodings_pred, decodings_true = _compute_cost_matrix(
        labels_pred=labels_pred, labels_true=labels_true, num_classes=num_classes, encode=encode
    )

    if cost_matrix.shape[0] == cost_matrix.shape[1]:
        row_ind, col_ind, _ = lapjv(-cost_matrix)
    else:
        row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    best_acc = cost_matrix[row_ind, col_ind].sum() / labels_pred.shape[0]
    assignments = {}
    for class_id, cluster_id in enumerate(col_ind):
        if decodings_true is not None:
            class_id = decodings_true[class_id]
        if decodings_pred is not None:
            cluster_id = decodings_pred[cluster_id]
        assignments[class_id] = cluster_id

    return best_acc, assignments


@numba.jit(nopython=True)
def _get_index_mapping(arr: np.ndarray) -> tuple[dict[int, int], dict[int, int]]:
    encodings, decodings = {}, {}
    for i, val in enumerate(np.unique(arr)):
        encodings[val] = i
        decodings[i] = val
    return encodings, decodings


def apply_encodiing_dict(arr: np.ndarray, encoding_dict: dict[int, int]) -> np.ndarray:
    return np.vectorize(encoding_dict.__getitem__)(arr)


def _compute_cost_matrix(
    labels_pred: np.ndarray,
    labels_true: np.ndarray,
    num_classes: int = True,
    encode: bool = True,
) -> tuple[np.ndarray, dict[int, int] | None, dict[int, int] | None]:
    if encode and num_classes is None:
        encodings_pred, decodings_pred = _get_index_mapping(labels_pred)
        encodings_true, decodings_true = _get_index_mapping(labels_true)
        labels_pred = apply_encodiing_dict(labels_pred, encodings_pred)
        labels_true = apply_encodiing_dict(labels_true, encodings_true)
        cost_matrix = np.zeros((len(encodings_true), len(encodings_pred)))
    else:
        if num_classes is None:
            cost_matrix = np.zeros((len(np.unique(labels_true)), len(np.unique(labels_pred))))
        else:
            cost_matrix = np.zeros((num_classes, num_classes))
        decodings_true, decodings_pred = None, None

    indices, counts = np.unique(np.stack([labels_true, labels_pred]), axis=1, return_counts=True)
    cost_matrix[tuple(indices)] += counts

    return cost_matrix, decodings_pred, decodings_true
