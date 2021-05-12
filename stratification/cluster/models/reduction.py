import warnings

from numba.core.errors import NumbaWarning
import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP

__all__ = ['HardnessAugmentedReducer', 'NoOpReducer', 'PCAReducer', 'UMAPReducer']


class Reducer:
    def __init__(self, **kwargs):
        raise NotImplementedError()

    def fit(self, X):
        raise NotImplementedError()

    def transform(self, X):
        raise NotImplementedError()

    def decrement_components(self):
        raise NotImplementedError()


class NoOpReducer(Reducer):
    """
    A no-op reduction method. Used when making changes using raw features.
    """

    def __init__(self, n_components=1, **kwargs):
        self.n_components = n_components

    def fit(self, X):
        return self

    def transform(self, X):
        return X

    def decrement_components(self):
        self.n_components -= 1


class PCAReducer(Reducer):
    """
    Simple wrapper for PCA.
    """

    def __init__(self, n_components=2, **kwargs):
        self.n_components = n_components
        self.model = PCA(n_components=n_components)

    def fit(self, X):
        self.model.fit(X)
        return self

    def transform(self, X):
        return self.model.transform(X)

    def decrement_components(self):
        self.n_components -= 1
        self.model.n_components -= 1


class UMAPReducer(Reducer):
    """
    Simple wrapper for UMAP, used for API consistency.
    """

    def __init__(self, n_components=2, **kwargs):
        self.n_components = n_components
        kwargs = {**{'n_neighbors': 10, 'min_dist': 0.0}, **kwargs}
        self.model = UMAP(n_components=n_components, **kwargs)

    def fit(self, X):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', NumbaWarning)
            self.model.fit(X)
        return self

    def transform(self, X):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', NumbaWarning)
            result = self.model.transform(X)
        return result

    def decrement_components(self):
        self.n_components -= 1
        self.model.n_components -= 1


class HardnessAugmentedReducer(Reducer):
    """
    A reducer that extracts the "hardness" component (i.e. component
    orthogonal to the decision boundary, for a binary classification model).
    Optionally takes in another reducer, whose components are appended to
    this hardness component (possibly with different weights).
    """

    def __init__(self, nn_model, base_reducer=None, hc_weight=1):
        if base_reducer is not None:
            base_reducer.decrement_components()
            if base_reducer.n_components == 0:
                base_reducer = None
        self.base_reducer = base_reducer
        self.fc = nn_model.module.fc if hasattr(nn_model, 'module') else nn_model.fc
        self.decision_bdy = (self.fc.weight[1] - self.fc.weight[0]).cpu().data.numpy()
        self.decision_bdy /= np.linalg.norm(self.decision_bdy)
        self.hc_weight = hc_weight

    def fit(self, X):
        hardness_scores = np.dot(X, self.decision_bdy)
        X = X - np.outer(hardness_scores, self.decision_bdy)
        if self.base_reducer:
            self.base_reducer.fit(X)
        return self

    def transform(self, X):
        hardness_scores = np.dot(X, self.decision_bdy)
        X1 = hardness_scores
        X1 = self.hc_weight * X1.reshape(len(X1), 1)
        if self.base_reducer:
            X2 = X - np.outer(hardness_scores, self.decision_bdy)
            X2 = self.base_reducer.transform(X2)
            return np.hstack((X1, X2))
        return X1
