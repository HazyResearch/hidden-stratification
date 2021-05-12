import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

DATA_SPLITS = ['train', 'train_clean', 'val', 'test']
LABEL_TYPES = ['superclass', 'subclass', 'true_subclass', 'alt_subclass']


class GEORGEDataset(Dataset):
    """
    Lightweight class that enforces design pattern used within the training
    loop. Essential components:

    split  (str)    must be in {'train', 'train_clean', 'val', 'test'}.
        - 'train' datasets are for model training. Data augmentation commonly applied.
        - 'train_clean' datasets are for model evaluation on the train set.
            Unshuffled and with no data augmentation. Used for clsutering step.
        - 'val' datasets are for model evaluation during training.
        - 'test' datasets are for model evaluation after training.

    X      (any)    input to model. Passed through directly.
    Y_dict (dict)   targets used for computing loss and metrics.
        - the 'superclass' key will be used for data quality loss computation
        - the 'subclass' key will be used to compute metrics, as well as for DRO loss
        - the 'true_subclass' key will be used to compute metrics, if available
    """

    def __init__(self, name, root, split, transform=None, download=False, ontology='default'):
        self.name = name
        self.root = root
        self.data_dir = os.path.join(self.root, name)
        self.split = split
        self.transform = transform
        assert self.split in DATA_SPLITS
        if not self._check_exists():
            if download:
                self._download()
            else:
                raise ValueError(f'{self.name} dataset not found.')

        self.ontology = ontology

        logging.info(f'Loading {self.split} split of {self.name}')
        self.X, self.Y_dict = self._load_samples()

        assert (
            'superclass' in self.Y_dict.keys()
        ), "Y_dict['superclass'] must be populated with superclass (target) labels."

        if 'true_subclass' in self.Y_dict.keys():
            logging.info('True subclass available.')
            self.true_subclass_available = True
        else:
            logging.info('True subclass unavailable.')
            self.true_subclass_available = False
        assert self.true_subclass_available

        sup_to_true_sub_map = build_sup_to_sub_map(
            self.Y_dict['superclass'], self.Y_dict['true_subclass']
        )
        self._class_maps = {'true_subclass': sup_to_true_sub_map}
        self._subclass_labels_added = False

    def _check_exists(self):
        """
        Checks if the dataset has been initialized.
        """
        raise NotImplementedError

    def _download(self):
        """
        Downloads the dataset if it could not be found
        """
        raise NotImplementedError

    def _load_samples(self):
        """
        Loads the X tensor and the Y_dict dictionary for training.
        """
        raise NotImplementedError

    def __len__(self):
        """
        Returns the length of the dataset by returning the length of
        one of the label lists.
        """
        return len(next(iter(self.Y_dict.values())))

    def __getitem__(self):
        """
        Must be overridden.
        """
        raise NotImplementedError

    def add_labels(self, key, values):
        """
        Adds a key-value pair to the labels dictionary.
        """
        assert len(values) == len(self)
        if key in self.Y_dict.keys():
            logging.info(f'{key} in Y_dict already exists and will be overwritten.')
        if isinstance(values, torch.Tensor):
            values_tensor = values.clone().detach()
        else:
            values_tensor = torch.tensor(values)
        self.Y_dict[key] = values_tensor

    def add_subclass_labels(self, subclass_labels, seed=0):
        """
        Adds subclass_labels. If subclass_labels is a string, it must be in
        {'superclass', 'true_subclass', 'random'}. Else, subclass_labels is a
        list of labels, and thus is added directly to Y_dict.
        """
        if type(subclass_labels) == str:
            # use a set of labels in Y_dict (commonly 'superclass' or 'true_subclass')
            if subclass_labels in self.Y_dict.keys():
                self.add_labels('subclass', self.Y_dict[subclass_labels])
            # use a set of random labels mimicking class proportions of 'true_subclass'
            elif subclass_labels == 'random':
                self.add_labels(
                    'subclass',
                    generate_random_labels(
                        self.Y_dict['superclass'], self.Y_dict['true_subclass'], seed=seed
                    ),
                )
            else:
                raise ValueError(f'subclass_labels string {subclass_labels} not recognized.')
        elif subclass_labels is not None:
            self.add_labels('subclass', subclass_labels)
        else:
            raise ValueError(f'subclass_labels object {subclass_labels} not recognized.')
        self._subclass_labels_added = True

    def get_num_classes(self, key):
        return torch.max(self.Y_dict[key]).item() + 1

    def get_labels(self, key):
        return self.Y_dict[key]

    def get_class_counts(self, key):
        class_map = (
            self.get_labels(key) == torch.arange(self.get_num_classes(key)).unsqueeze(1).long()
        )
        return class_map.sum(1).float()

    def get_class_map(self, key):
        if key in self._class_maps:
            return self._class_maps[key]
        else:
            assert self._subclass_labels_added
            sup_to_sub_map = build_sup_to_sub_map(self.Y_dict['superclass'], self.Y_dict[key])
            self._class_maps[key] = sup_to_sub_map
            return sup_to_sub_map


def build_sup_to_sub_map(superclass_labels, subclass_labels):
    class_map = {}
    superclass_set = sorted(set(np.array(superclass_labels)))
    for superclass in superclass_set:
        class_map[superclass] = sorted(
            np.unique(np.array(subclass_labels[superclass_labels == superclass]))
        )
    return class_map


def generate_random_labels(superclass_labels, subclass_labels, proportions=None, seed=0):
    """
    Build random mock subclass labels for each superclass, with the given proportions.
    If proportions is None, uses the proportions of the given subclass labels.
    """
    prev_state = random.getstate()
    random.seed(seed)
    data_mod_seed = random.randint(0, 2 ** 32)
    random.seed(data_mod_seed)

    superclass_labels, subclass_labels = np.array(superclass_labels), np.array(subclass_labels)
    random_labels = -np.ones_like(superclass_labels)
    superclass_set = sorted(set(superclass_labels))
    if proportions is None:
        proportions = []
        sup_to_sub_map = build_sup_to_sub_map(superclass_labels, subclass_labels)
        for superclass in superclass_set:
            proportions.append([])
            for subclass in sup_to_sub_map[superclass]:
                superclass_indices = superclass_labels == superclass
                # Calculate the proportion of examples of this superclass that are of this subclass
                proportions[superclass].append(
                    np.mean(subclass_labels[superclass_indices] == subclass)
                )
    for superclass in superclass_set:
        superclass_indices = superclass_labels == superclass
        num_subclass_examples = np.sum(superclass_indices)
        subclass_proportions = proportions[superclass]
        cumulative_prop = np.cumsum(subclass_proportions)
        cumulative_prop = np.round(cumulative_prop * num_subclass_examples).astype(np.int)
        cumulative_prop = np.concatenate(([0], cumulative_prop))
        assert cumulative_prop[-1] == num_subclass_examples
        mock_sub = -np.ones(num_subclass_examples)
        for i in range(len(cumulative_prop) - 1):
            percentile_lower, percentile_upper = cumulative_prop[i], cumulative_prop[i + 1]
            mock_sub[percentile_lower:percentile_upper] = i
        assert np.all(mock_sub >= 0)
        mock_sub = mock_sub + np.amax(random_labels) + 1  # adjust for previous superclasses
        random.shuffle(mock_sub)
        random_labels[superclass_indices] = mock_sub
    assert np.all(random_labels >= 0)

    random.setstate(prev_state)
    return torch.tensor(random_labels)
