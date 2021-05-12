try:
    from libKMCUDA import kmeans_cuda

    _LIBKMCUDA_FOUND = True
except ModuleNotFoundError:
    _LIBKMCUDA_FOUND = False

from functools import partial
import logging

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from stratification.cluster.utils import silhouette_samples

__all__ = [
    'KMeans',
    'GaussianMixture',
    'FastKMeans',
    'AutoKMixtureModel',
    'OverclusterModel',
    'DummyClusterer',
]


def get_cluster_sils(data, pred_labels, compute_sil=True, cuda=False):
    unique_preds = sorted(np.unique(pred_labels))
    SIL_samples = (
        silhouette_samples(data, pred_labels, cuda=cuda) if compute_sil else np.zeros(len(data))
    )
    SILs_by_cluster = {
        int(label): float(np.mean(SIL_samples[pred_labels == label])) for label in unique_preds
    }
    SIL_global = float(np.mean(SIL_samples))
    return SILs_by_cluster, SIL_global


def compute_group_sizes(labels):
    result = dict(sorted(zip(*np.unique(labels, return_counts=True))))
    return {int(k): int(v) for k, v in result.items()}


class DummyClusterer:
    def __init__(self, **kwargs):
        self.n_components = 1

    def fit(self, X):
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=np.int32)


class FastKMeans:
    def __init__(self, n_clusters, random_state=0, init='k-means++', n_init=10, verbose=False):
        self.k = n_clusters
        self.init = init
        if n_init > 1:
            logging.warning('n_init unsupported for GPU K-Means')
        self.seed = random_state
        self.verbose = verbose
        self.kmeans_obj = KMeans(n_clusters=n_clusters)

    def fit(self, X):
        logging.info('Using GPU-accelerated K-Means...')
        self.cluster_centers_ = kmeans_cuda(
            X.astype(np.float32), clusters=self.k, seed=self.seed, init=self.init
        )[0].astype(np.float32)
        self.kmeans_obj.cluster_centers_ = self.cluster_centers_
        if hasattr(self.kmeans_obj, '_check_params'):
            self.kmeans_obj._check_params(np.zeros_like(X))  # properly initialize
        return self.kmeans_obj

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    def predict(self, X):
        return self.kmeans_obj.predict(X.astype(np.float32))

    def transform(self, X):
        return self.kmeans_obj.transform(X.astype(np.float32))


class AutoKMixtureModel:
    def __init__(
        self, cluster_method, max_k, n_init=3, seed=None, sil_cuda=False, verbose=0, search=True
    ):
        if cluster_method == 'kmeans':
            cluster_cls = FastKMeans if (sil_cuda and _LIBKMCUDA_FOUND) else KMeans
            k_name = 'n_clusters'
        elif cluster_method == 'gmm':
            cluster_cls = GaussianMixture
            k_name = 'n_components'
        else:
            raise ValueError('Unsupported clustering method')

        self.cluster_cls = cluster_cls
        self.k_name = k_name
        self.search = search
        self.max_k = max_k
        self.n_init = n_init
        self.seed = seed
        self.sil_cuda = sil_cuda
        self.verbose = verbose

    def gen_inner_cluster_obj(self, k):
        # Return a clustering object according to the specified parameters
        return self.cluster_cls(
            **{self.k_name: k}, n_init=self.n_init, random_state=self.seed, verbose=self.verbose
        )

    def fit(self, activ):
        logger = logging.getLogger('harness.cluster')
        best_score = -2
        k_min = 2 if self.search else self.max_k
        search = self.search and k_min != self.max_k
        for k in range(k_min, self.max_k + 1):
            logger.info(f'Clustering into {k} groups...')
            cluster_obj = self.gen_inner_cluster_obj(k)
            pred_labels = cluster_obj.fit_predict(activ)
            logger.info('Clustering done, computing score...')
            cluster_sizes = compute_group_sizes(pred_labels)
            if search:
                local_sils, global_sil = get_cluster_sils(
                    activ, pred_labels, compute_sil=True, cuda=self.sil_cuda
                )
                clustering_score = np.mean(list(local_sils.values()))
                logger.info(f'k = {k} score: {clustering_score}')
                if clustering_score >= best_score:
                    logger.info(f'Best model found at k = {k} with score {clustering_score:.3f}')
                    best_score = clustering_score
                    best_model = cluster_obj
                    best_k = k
            else:
                best_score, best_model, best_k = 0, cluster_obj, self.max_k

        self.best_k = best_k
        self.n_clusters = best_k
        self.best_score = best_score
        self.cluster_obj = best_model
        return self

    def predict(self, activ):
        return self.cluster_obj.predict(activ)

    def fit_predict(self, activ):
        self.fit(activ)
        return self.predict(activ)

    def predict_proba(self, X):
        return self.cluster_obj.predict_proba(activ)

    def score(self, X):
        return self.cluster_obj.score(activ)


class OverclusterModel:
    def __init__(
        self,
        cluster_method,
        max_k,
        oc_fac,
        n_init=3,
        search=True,
        sil_threshold=0.0,
        seed=None,
        sil_cuda=False,
        verbose=0,
        sz_threshold_pct=0.005,
        sz_threshold_abs=25,
    ):
        self.base_model = AutoKMixtureModel(
            cluster_method, max_k, n_init, seed, sil_cuda, verbose, search
        )
        self.oc_fac = oc_fac
        self.sil_threshold = sil_threshold
        self.sz_threshold_pct = sz_threshold_pct
        self.sz_threshold_abs = sz_threshold_abs
        self.requires_extra_info = True

    def get_oc_predictions(self, activ, val_activ, orig_preds, val_orig_preds):
        # Split each cluster from base_model into sub-clusters, and save each of the
        # associated sub-clustering predictors in self.cluster_objs.
        # Collate and return the new predictions in oc_preds and val_oc_preds.
        self.cluster_objs = []
        oc_preds = np.zeros(len(activ), dtype=np.int)
        val_oc_preds = np.zeros(len(val_activ), dtype=np.int)

        for i in self.pred_vals:
            sub_activ = activ[orig_preds == i]
            cluster_obj = self.base_model.gen_inner_cluster_obj(self.oc_fac).fit(sub_activ)
            self.cluster_objs.append(cluster_obj)
            sub_preds = cluster_obj.predict(sub_activ) + self.oc_fac * i
            oc_preds[orig_preds == i] = sub_preds

            val_sub_activ = val_activ[val_orig_preds == i]
            val_sub_preds = cluster_obj.predict(val_sub_activ) + self.oc_fac * i
            val_oc_preds[val_orig_preds == i] = val_sub_preds
        return oc_preds, val_oc_preds

    def filter_overclusters(self, activ, losses, orig_preds, oc_preds, val_oc_preds):
        # Keep an overcluster if its point have higher SIL than before
        # overclustering, AND it has higher average loss than the
        # original cluster, AND it contains sufficiently many training and
        # validation points.

        num_oc = np.amax(oc_preds) + 1
        # Compute original per-cluster SIL scores and losses,
        # and the SIL scores and losses after overclustering.
        orig_sample_sils = silhouette_samples(activ, orig_preds, cuda=self.sil_cuda)
        orig_losses = [np.mean(losses[orig_preds == i]) for i in self.pred_vals]
        new_sample_sils = silhouette_samples(activ, oc_preds, cuda=self.sil_cuda)

        oc_orig_sils = [np.mean(orig_sample_sils[oc_preds == i]) for i in range(num_oc)]
        oc_new_sils = [np.mean(new_sample_sils[oc_preds == i]) for i in range(num_oc)]
        new_losses = [np.mean(losses[oc_preds == i]) for i in range(num_oc)]

        # Count number of points in each cluster after overclustering. Drop tiny clusters as these
        # will lead to unreliable optimization.
        oc_counts = np.bincount(oc_preds)
        # If val clusters are too small, we will get unreliable estimates - so need to threshold these too
        val_oc_counts = np.bincount(val_oc_preds)
        tr_sz_threshold = max(len(activ) * self.sz_threshold_pct, self.sz_threshold_abs)
        val_sz_threshold = self.sz_threshold_abs

        # Decide which overclusters to keep
        oc_to_keep = []
        for i in range(num_oc):
            if (
                oc_new_sils[i] > max(oc_orig_sils[i], self.sil_threshold)
                and new_losses[i] >= orig_losses[i // self.oc_fac]
                and oc_counts[i] >= tr_sz_threshold
                and val_oc_counts[i] >= val_sz_threshold
            ):
                oc_to_keep.append(i)

        return oc_to_keep

    def create_label_map(self, num_orig_preds, oc_to_keep, oc_preds):
        # Map raw overclustering outputs to final "cluster labels," accounting for the
        # fact that some overclusters are re-merged.
        label_map = {}
        cur_cluster_ind = -1
        oc_to_base_id = {}
        for i in range(num_orig_preds):
            # For each original cluster, if there were no
            # overclusters kept within it, keep the original cluster as-is.
            # Otherwise, it needs to be split.
            keep_all = True  # If we keep all overclusters, we can discard the original cluster
            for j in range(self.oc_fac):
                index = i * self.oc_fac + j
                if index not in oc_to_keep:
                    keep_all = False
            if not keep_all:
                cur_cluster_ind += 1

            # Updated cluster index corresponding to original cluster
            # (points in the original cluster assigned to a non-kept overcluster
            # are merged into this cluster)
            base_index = cur_cluster_ind
            for j in range(self.oc_fac):
                index = i * self.oc_fac + j
                if index in oc_to_keep:
                    cur_cluster_ind += 1
                    oc_index = cur_cluster_ind
                else:
                    assert not keep_all
                    oc_index = base_index
                label_map[index] = oc_index
        return label_map

    def fit(self, activ, val_activ=None, losses=None):
        if val_activ is None or losses is None:
            raise ValueError('Must provide losses and val set activations')
        logger = logging.getLogger('harness.cluster')
        logger.info('Fitting base model...')
        orig_preds = self.base_model.fit_predict(activ)
        self.pred_vals = sorted(np.unique(orig_preds))
        num_orig_preds = len(self.pred_vals)
        losses = np.array(losses)
        oc_fac = self.oc_fac
        num_oc = num_orig_preds * oc_fac
        val_orig_preds = self.base_model.predict(val_activ)

        logger.info('Fitting overclustering model...')
        oc_preds, val_oc_preds = self.get_oc_predictions(
            activ, val_activ, orig_preds, val_orig_preds
        )
        oc_to_keep = self.filter_overclusters(activ, losses, orig_preds, oc_preds, val_oc_preds)
        self.label_map = self.create_label_map(num_orig_preds, oc_to_keep, oc_preds)

        new_preds = np.zeros(len(activ), dtype=np.int)
        for i in range(num_oc):
            new_preds[oc_preds == i] = self.label_map[i]

        self.n_clusters = max(self.label_map.values()) + 1  # Final number of output predictions
        logger.info(f'Final number of clusters: {self.n_clusters}')
        return self

    def predict(self, activ):
        # Get clusters from base model
        base_preds = self.base_model.predict(activ)
        # Get overclusters
        oc_preds = np.zeros(len(activ), dtype=np.int)
        for i in self.pred_vals:
            subfeats = activ[base_preds == i]
            subpreds = self.cluster_objs[i].predict(subfeats) + self.oc_fac * i
            oc_preds[base_preds == i] = subpreds

        # Merge overclusters appropriately and return final predictions
        new_preds = np.zeros(len(activ), dtype=np.int)
        for i in range(len(self.pred_vals) * self.oc_fac):
            new_preds[oc_preds == i] = self.label_map[i]
        return new_preds

    @property
    def sil_cuda(self):
        return self.base_model.sil_cuda

    @property
    def n_init(self):
        return self.base_model.n_init

    @property
    def seed(self):
        return self.base_model.seed
