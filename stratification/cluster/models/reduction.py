from numba.core.errors import NumbaWarning
from sklearn.decomposition import PCA
from umap import UMAP
import warnings


class NoOpReducer:
    """
    A no-op reduction method. Used when making changes using raw features.
    """
    def __init__(self, **kwargs):
        pass

    def fit(self, X):
        return self

    def transform(self, X):
        return X


class UMAPReducer:
    def __init__(self, **kwargs):
        self.model = UMAP(**kwargs)

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
