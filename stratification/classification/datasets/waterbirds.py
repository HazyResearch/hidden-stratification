from collections import Counter
import itertools
import logging
import os

from PIL import Image
import numpy as np
import pandas as pd
import torch
from torchvision import transforms

from stratification.classification.datasets.base import GEORGEDataset


class WaterbirdsDataset(GEORGEDataset):
    """Waterbirds Dataset"""

    _channels = 3
    _resolution = 224
    _normalization_stats = {'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)}
    # used to determine subclasses (index here used for querying sample class)
    _df_attr_keys = ['y', 'place']
    split_dict = {'train': 0, 'val': 1, 'test': 2}

    def __init__(
        self, root, split, transform=None, download=False, ontology='default', augment=False
    ):
        assert transform is None
        transform = get_transform_cub()
        super().__init__(
            'waterbirds', root, split, transform=transform, download=download, ontology=ontology
        )

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'waterbirds')

    def _check_exists(self):
        """Checks whether or not the waterbirds labels CSV has been initialized."""
        return os.path.isfile(os.path.join(self.processed_folder, 'metadata.csv'))

    def _download(self):
        """Raises an error if the raw dataset has not yet been downloaded."""
        raise ValueError('Follow the README instructions to download the dataset.')

    def _load_samples(self):
        """Loads the Waterbirds dataset"""
        self.df = pd.read_csv(os.path.join(self.processed_folder, 'metadata.csv'))

        # initialize the subclasses (before split because should be invariant across splits)
        class_attrs_to_class_idx = self._get_class_attrs_to_class_idx()

        # split dataset
        split_df = self.df[self.df['split'] == self.split_dict[self.split]]
        logging.info(f'Using {len(split_df)} of {len(self.df)} images in split "{self.split}"')
        self.df = split_df

        # gets the data
        image_names, original_labels = self._get_data(class_attrs_to_class_idx)

        # reverse dict to easily lookup class_attrs from label
        class_idx_to_class_attrs = {i: k for k, i in class_attrs_to_class_idx.items()}
        logging.info('Sample attributes:')
        logging.info(self._df_attr_keys)
        logging.info('--')
        logging.info(f'Original label counts ({self.split} split):')
        logging.info(
            '\n'.join(
                [
                    f'idx: {class_idx},\t count: {class_count},\t '
                    f'attrs:{class_idx_to_class_attrs[class_idx]}'
                    for class_idx, class_count in sorted(Counter(original_labels).items())
                ]
            )
        )
        logging.info('--')
        superclass_labels, self.superclass_names = self._get_superclass_labels_from_id(
            original_labels, class_idx_to_class_attrs
        )
        true_subclass_labels, self.true_subclass_names = self._get_true_subclass_labels_from_id(
            original_labels, class_idx_to_class_attrs
        )
        X = image_names
        Y_dict = {
            'superclass': torch.from_numpy(superclass_labels),
            'true_subclass': torch.from_numpy(true_subclass_labels),
            'original': torch.from_numpy(original_labels),
        }
        return X, Y_dict

    def _get_data(self, class_attrs_to_class_idx):
        """
        Iterates through the DataFrame to extract the image name and label.

        The subclass labels are automatically assigned based on the row's attributes.
        """
        image_names = []
        labels = []
        for idx, row in self.df.iterrows():
            image_name = row['img_filename']
            image_names.append(image_name)
            row_attrs = []
            for df_attr_key in self._df_attr_keys:
                row_attr = row[df_attr_key]
                row_attrs.append(row_attr)
            label = class_attrs_to_class_idx[tuple(row_attrs)]
            labels.append(label)
        image_names = np.array(image_names)
        labels = np.array(labels)
        return image_names, labels

    def _get_class_attrs_to_class_idx(self):
        """Uses self._df_attr_keys to identify all possible subclasses.

        Subclass labels (class_idx) are mapped to a tuple of sample attributes (class_attrs).
        """
        df_attr_uniques = []
        for i, df_attr_key in enumerate(self._df_attr_keys):
            uniques = sorted(self.df[df_attr_key].unique())
            df_attr_uniques.append(uniques)
        class_attrs_to_class_idx = {k: i for i, k in enumerate(itertools.product(*df_attr_uniques))}
        return class_attrs_to_class_idx

    def _get_superclass_labels_from_id(self, original_labels, class_idx_to_class_attrs):
        """Superclass labels are determined from the original_labels by the given ontology.

        The default"""
        superclass_labels = []
        if self.ontology == 'default':
            y_attr_idx = self._df_attr_keys.index('y')
            for label in original_labels:
                class_attrs = class_idx_to_class_attrs[label]
                if class_attrs[y_attr_idx] == 0:
                    superclass_label = 0
                elif class_attrs[y_attr_idx] == 1:
                    superclass_label = 1
                else:
                    raise ValueError(
                        f'Unrecognized class attributes {class_attrs} for label {label}'
                    )
                superclass_labels.append(superclass_label)
            superclass_names = ['LANDBIRD', 'WATERBIRD']
        else:
            raise ValueError(f'superclass id {self.ontology} not recognized.')
        return np.array(superclass_labels), superclass_names

    def _get_true_subclass_labels_from_id(self, original_labels, class_idx_to_class_attrs):
        """True subclass labels are determined from the original_labels by the given ontology"""
        true_subclass_labels = []
        if self.ontology == 'default':
            y_attr_idx = self._df_attr_keys.index('y')
            place_attr_idx = self._df_attr_keys.index('place')
            for label in original_labels:
                class_attrs = class_idx_to_class_attrs[label]
                if class_attrs[y_attr_idx] == 0 and class_attrs[place_attr_idx] == 0:
                    true_subclass_label = 0
                elif class_attrs[y_attr_idx] == 0 and class_attrs[place_attr_idx] == 1:
                    true_subclass_label = 1
                elif class_attrs[y_attr_idx] == 1 and class_attrs[place_attr_idx] == 0:
                    true_subclass_label = 2
                elif class_attrs[y_attr_idx] == 1 and class_attrs[place_attr_idx] == 1:
                    true_subclass_label = 3
                else:
                    raise ValueError(
                        f'Unrecognized class attributes {class_attrs} for label {label}'
                    )
                true_subclass_labels.append(true_subclass_label)
            true_subclass_names = [
                'LANDBIRD on land',
                'LANDBIRD on water',
                'WATERBIRD on land',
                'WATERBIRD on water',
            ]
        else:
            raise ValueError(f'subclass id {self.ontology} not recognized.')
        return np.array(true_subclass_labels), true_subclass_names

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (x: torch.Tensor, y: dict) where X is a tensor representing an image
                and y is a dictionary of possible labels.
        """
        image_path = os.path.join(self.data_dir, self.X[idx])
        image = Image.open(image_path)

        if self.transform is not None:
            image = self.transform(image)
        x = image

        y_dict = {name: label[idx] for name, label in self.Y_dict.items()}
        return x, y_dict


def get_transform_cub():
    target_resolution = (224, 224)

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize(**WaterbirdsDataset._normalization_stats),
        ]
    )
    return transform
