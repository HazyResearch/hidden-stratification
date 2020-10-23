import os
import json
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from copy import deepcopy

from stratification.harness import GEORGEHarness
from stratification.classification.models import ShallowCNN
from stratification.classification.datasets import MNISTDataset, DATA_SPLITS
from stratification.cluster.models.cluster import GaussianMixture
from stratification.cluster.models.reduction import UMAPReducer
from stratification.utils.utils import set_seed
from stratification.utils.parse_args import get_config


def main():
    config = get_config()
    use_cuda_if_available = False  # change to True if you want to use CUDA
    use_cuda = use_cuda_if_available and torch.cuda.is_available()
    set_seed(config['seed'], use_cuda)  # set seeds for reproducibility

    dataset_class = MNISTDataset

    # Use mild data augmentation during training
    transform_train = transforms.Compose([
        transforms.RandomCrop(dataset_class._resolution, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**dataset_class._normalization_stats)
    ])
    transform_eval = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(**dataset_class._normalization_stats)])

    dataloaders = {}
    batch_size = config['classification_config']['batch_size']
    for split in DATA_SPLITS:
        if split == 'train':
            # Training dataloader uses transform_train transforms, and is shuffled
            dataset = dataset_class(root='./data', split=split, transform=transform_train,
                                    download=True)
            dataloaders[split] = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=8)
        else:
            # Evaluation dataloaders (including for the training set) are "clean" - no data augmentation or shuffling
            key = 'train' if split == 'train_clean' else split
            dataset = dataset_class(root='./data', split=key, transform=transform_eval)
            dataloaders[split] = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=8)

    model = ShallowCNN(num_classes=2)
    print('Model architecture:')
    print(model)

    # specify layer for which to save activations
    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
        print('Using CUDA')
    state_dict = deepcopy(model.state_dict())

    torch.multiprocessing.set_sharing_strategy('file_system')
    harness = GEORGEHarness(config['exp_dir'], use_cuda=use_cuda)

    # Train a model with ERM
    erm_dir = harness.classify(config['classification_config'], model, dataloaders, 'erm')

    # Cluster the activations of the model
    reduction_model = UMAPReducer(random_state=12345, n_components=2, n_neighbors=10, min_dist=0)
    cluster_model = GaussianMixture(covariance_type='full', n_components=5, n_init=3)
    cluster_dir = harness.cluster(config['cluster_config'], cluster_model,
                                  inputs_path=os.path.join(erm_dir, 'outputs.pt'),
                                  reduction_model=reduction_model)

    model.load_state_dict(state_dict)  # reset model state
    # Train the final (GEORGE) model
    george_dir = harness.classify(config['classification_config'], model, dataloaders, 'george',
                                  clusters_path=os.path.join(cluster_dir, 'clusters.pt'))


if __name__ == '__main__':
    main()
