'''The functions in this file are adapted from scikit-learn
(https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/cluster/_unsupervised.py)
to use CUDA for Silhouette score computation.'''

import numpy as np
from sklearn.metrics import silhouette_samples as s_sil
from sklearn.metrics.cluster._unsupervised import *
from sklearn.utils import gen_batches, get_chunk_n_rows
import torch


def silhouette_samples(X, labels, verbose=False, cuda=False):
    if not cuda:
        return s_sil(X, labels)
    X, labels = check_X_y(X, labels, accept_sparse=['csc', 'csr'])

    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples = len(labels)
    label_freqs = np.bincount(labels)
    check_number_of_labels(len(le.classes_), n_samples)

    reduce_func = functools.partial(_silhouette_reduce, labels=labels, label_freqs=label_freqs)
    results = zip(*pairwise_distances_chunked_cuda(X, reduce_func=reduce_func, verbose=verbose))
    intra_clust_dists, inter_clust_dists = results
    intra_clust_dists = np.concatenate(intra_clust_dists)
    inter_clust_dists = np.concatenate(inter_clust_dists)

    denom = (label_freqs - 1).take(labels, mode='clip')
    with np.errstate(divide="ignore", invalid="ignore"):
        intra_clust_dists /= denom

    sil_samples = inter_clust_dists - intra_clust_dists
    with np.errstate(divide="ignore", invalid="ignore"):
        sil_samples /= np.maximum(intra_clust_dists, inter_clust_dists)
    # nan values are for clusters of size 1, and should be 0
    return np.nan_to_num(sil_samples)


def _silhouette_reduce(D_chunk, start, labels, label_freqs):
    """Accumulate silhouette statistics for vertical chunk of X
    Parameters
    ----------
    D_chunk : shape (n_chunk_samples, n_samples)
        precomputed distances for a chunk
    start : int
        first index in chunk
    labels : array, shape (n_samples,)
        corresponding cluster labels, encoded as {0, ..., n_clusters-1}
    label_freqs : array
        distribution of cluster labels in ``labels``
    """
    # accumulate distances from each sample to each cluster
    clust_dists = np.zeros((len(D_chunk), len(label_freqs)), dtype=D_chunk.dtype)
    for i in range(len(D_chunk)):
        clust_dists[i] += np.bincount(labels, weights=D_chunk[i], minlength=len(label_freqs))

    # intra_index selects intra-cluster distances within clust_dists
    intra_index = (np.arange(len(D_chunk)), labels[start : start + len(D_chunk)])
    # intra_clust_dists are averaged over cluster size outside this function
    intra_clust_dists = clust_dists[intra_index]
    # of the remaining distances we normalise and extract the minimum
    clust_dists[intra_index] = np.inf
    clust_dists /= label_freqs
    inter_clust_dists = clust_dists.min(axis=1)
    return intra_clust_dists, inter_clust_dists


def _check_chunk_size(reduced, chunk_size):
    """Checks chunk is a sequence of expected size or a tuple of same"""
    if reduced is None:
        return
    is_tuple = isinstance(reduced, tuple)
    if not is_tuple:
        reduced = (reduced,)
    if any(isinstance(r, tuple) or not hasattr(r, '__iter__') for r in reduced):
        raise TypeError(
            'reduce_func returned %r. '
            'Expected sequence(s) of length %d.' % (reduced if is_tuple else reduced[0], chunk_size)
        )
    if any(len(r) != chunk_size for r in reduced):
        actual_size = tuple(len(r) for r in reduced)
        raise ValueError(
            'reduce_func returned object of length %s. '
            'Expected same length as input: %d.'
            % (actual_size if is_tuple else actual_size[0], chunk_size)
        )


def pairwise_distances_chunked_cuda(X, reduce_func=None, verbose=False):
    """Generate a distance matrix chunk by chunk with optional reduction
    In cases where not all of a pairwise distance matrix needs to be stored at
    once, this is used to calculate pairwise distances in
    ``working_memory``-sized chunks.  If ``reduce_func`` is given, it is run
    on each chunk and its return values are concatenated into lists, arrays
    or sparse matrices.
    Parameters
    ----------
    X : array [n_samples_a, n_samples_a] if metric == "precomputed", or,
        [n_samples_a, n_features] otherwise
        Array of pairwise distances between samples, or a feature array.
    Y : array [n_samples_b, n_features], optional
        An optional second feature array. Only allowed if
        metric != "precomputed".
    reduce_func : callable, optional
        The function which is applied on each chunk of the distance matrix,
        reducing it to needed values.  ``reduce_func(D_chunk, start)``
        is called repeatedly, where ``D_chunk`` is a contiguous vertical
        slice of the pairwise distance matrix, starting at row ``start``.
        It should return one of: None; an array, a list, or a sparse matrix
        of length ``D_chunk.shape[0]``; or a tuple of such objects. Returning
        None is useful for in-place operations, rather than reductions.
        If None, pairwise_distances_chunked returns a generator of vertical
        chunks of the distance matrix.
    metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by scipy.spatial.distance.pdist for its metric parameter, or
        a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
        If metric is "precomputed", X is assumed to be a distance matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from X as input and return a value indicating
        the distance between them.
    n_jobs : int or None, optional (default=None)
        The number of jobs to use for the computation. This works by breaking
        down the pairwise matrix into n_jobs even slices and computing them in
        parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    working_memory : int, optional
        The sought maximum memory for temporary distance matrix chunks.
        When None (default), the value of
        ``sklearn.get_config()['working_memory']`` is used.
    `**kwds` : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a scipy.spatial.distance metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.
    Yields
    ------
    D_chunk : array or sparse matrix
        A contiguous slice of distance matrix, optionally processed by
        ``reduce_func``.
    Examples
    --------
    Without reduce_func:
    >>> import numpy as np
    >>> from sklearn.metrics import pairwise_distances_chunked
    >>> X = np.random.RandomState(0).rand(5, 3)
    >>> D_chunk = next(pairwise_distances_chunked(X))
    >>> D_chunk
    array([[0.  ..., 0.29..., 0.41..., 0.19..., 0.57...],
           [0.29..., 0.  ..., 0.57..., 0.41..., 0.76...],
           [0.41..., 0.57..., 0.  ..., 0.44..., 0.90...],
           [0.19..., 0.41..., 0.44..., 0.  ..., 0.51...],
           [0.57..., 0.76..., 0.90..., 0.51..., 0.  ...]])
    Retrieve all neighbors and average distance within radius r:
    >>> r = .2
    >>> def reduce_func(D_chunk, start):
    ...     neigh = [np.flatnonzero(d < r) for d in D_chunk]
    ...     avg_dist = (D_chunk * (D_chunk < r)).mean(axis=1)
    ...     return neigh, avg_dist
    >>> gen = pairwise_distances_chunked(X, reduce_func=reduce_func)
    >>> neigh, avg_dist = next(gen)
    >>> neigh
    [array([0, 3]), array([1]), array([2]), array([0, 3]), array([4])]
    >>> avg_dist
    array([0.039..., 0.        , 0.        , 0.039..., 0.        ])
    Where r is defined per sample, we need to make use of ``start``:
    >>> r = [.2, .4, .4, .3, .1]
    >>> def reduce_func(D_chunk, start):
    ...     neigh = [np.flatnonzero(d < r[i])
    ...              for i, d in enumerate(D_chunk, start)]
    ...     return neigh
    >>> neigh = next(pairwise_distances_chunked(X, reduce_func=reduce_func))
    >>> neigh
    [array([0, 3]), array([0, 1]), array([2]), array([0, 3]), array([4])]
    Force row-by-row generation by reducing ``working_memory``:
    >>> gen = pairwise_distances_chunked(X, reduce_func=reduce_func,
    ...                                  working_memory=0)
    >>> next(gen)
    [array([0, 3])]
    >>> next(gen)
    [array([0, 1])]
    """
    X = X.astype(np.float32)
    n_samples_X = len(X)
    Y = X
    # We get as many rows as possible within our working_memory budget to
    # store len(Y) distances in each row of output.
    #
    # Note:
    #  - this will get at least 1 row, even if 1 row of distances will
    #    exceed working_memory.
    #  - this does not account for any temporary memory usage while
    #    calculating distances (e.g. difference of vectors in manhattan
    #    distance.
    chunk_n_rows = get_chunk_n_rows(
        row_bytes=8 * len(Y), max_n_rows=n_samples_X, working_memory=None
    )
    slices = gen_batches(n_samples_X, chunk_n_rows)

    X_full = torch.tensor(X).cuda()
    Xnorms = torch.norm(X_full, dim=1, keepdim=True) ** 2
    for sl in slices:
        if verbose:
            print(sl)
        if sl.start == 0 and sl.stop == n_samples_X:
            X_chunk = X  # enable optimised paths for X is Y
        else:
            X_chunk = X[sl]
        pX = torch.tensor(X_chunk).cuda()
        d2 = Xnorms[sl] - 2 * torch.matmul(pX, X_full.t()) + Xnorms.t()
        d2 = torch.sqrt(torch.nn.functional.relu(d2)).cpu().numpy()
        d2.flat[sl.start :: len(X) + 1] = 0
        D_chunk = d2
        if reduce_func is not None:
            chunk_size = D_chunk.shape[0]
            D_chunk = reduce_func(D_chunk, sl.start)
            _check_chunk_size(D_chunk, chunk_size)
        yield D_chunk
