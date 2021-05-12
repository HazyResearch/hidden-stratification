import os

import torch

from stratification.cluster.models.cluster import GaussianMixture
from stratification.cluster.models.reduction import UMAPReducer
from stratification.harness import GEORGEHarness
from stratification.utils.parse_args import get_config
from stratification.utils.utils import init_cuda, set_seed


def main():
    config = get_config()
    use_cuda = config['use_cuda'] and torch.cuda.is_available()
    set_seed(config['seed'], use_cuda)  # set seeds for reproducibility
    init_cuda(config['deterministic'], config['allow_multigpu'])

    torch.multiprocessing.set_sharing_strategy('file_system')
    harness = GEORGEHarness(config, use_cuda=use_cuda)
    harness.save_full_config(config)

    dataloaders = harness.get_dataloaders(config, mode='erm')
    num_classes = dataloaders['train'].dataset.get_num_classes('superclass')
    model = harness.get_nn_model(config, num_classes=num_classes, mode='erm')

    print('Model architecture:')
    print(model)

    # Train a model with ERM
    erm_dir = harness.classify(config['classification_config'], model, dataloaders, 'erm')

    # Cluster the activations of the model
    reduction_model = UMAPReducer(random_state=12345, n_components=2, n_neighbors=10, min_dist=0)
    reduction_dir = harness.reduce(
        config['reduction_config'], reduction_model, inputs_path=os.path.join(erm_dir, 'outputs.pt')
    )
    cluster_model = GaussianMixture(covariance_type='full', n_components=5, n_init=3)
    cluster_dir = harness.cluster(
        config['cluster_config'],
        cluster_model,
        inputs_path=os.path.join(reduction_dir, 'outputs.pt'),
    )

    set_seed(config['seed'], use_cuda)  # reset random state
    dataloaders = harness.get_dataloaders(
        config, mode='george', subclass_labels=os.path.join(cluster_dir, 'clusters.pt')
    )
    model = harness.get_nn_model(config, num_classes=num_classes, mode='george')

    # Train the final (GEORGE) model
    george_dir = harness.classify(
        config['classification_config'], model, dataloaders, mode='george'
    )


if __name__ == '__main__':
    main()
