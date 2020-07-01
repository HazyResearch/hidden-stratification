from collections import Counter
import numpy as np


def get_k_from_model(model):
    if hasattr(model, 'n_clusters'):
        return model.n_clusters
    elif hasattr(model, 'n_components'):
        return model.n_components
    else:
        raise NotImplementedError(f'model {type(model)} K not found.' +
                                  f'model attributes:\n{list(model.__dict__.keys())}')


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
