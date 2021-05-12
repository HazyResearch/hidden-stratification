import codecs
from collections import defaultdict
import logging
import os
import random

from PIL import Image
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive

from .base import GEORGEDataset


class MNISTDataset(GEORGEDataset):
    """MNIST Dataset, possibly with undersampling.

    NOTE: creates validation set when downloaded for the first time.
    This is a deviation from the traditional MNIST dataset setup.

    See <https://pytorch.org/docs/stable/_modules/torchvision/datasets/mnist.html>.
    """

    resources = [
        (
            'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'f68b3c2dcbeaaa9fbdd348bbdeb94873',
        ),
        (
            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
            'd53e105ee54ea40749a09fcbcd1e9432',
        ),
        (
            'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            '9fb629c4189551a2d022fa330f9573f3',
        ),
        (
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
            'ec29112dd5afa0611ce80d1b7f02629c',
        ),
    ]
    true_subclass_names = [
        '0 - zero',
        '1 - one',
        '2 - two',
        '3 - three',
        '4 - four',
        '5 - five',
        '6 - six',
        '7 - seven',
        '8 - eight',
        '9 - nine',
    ]
    _channels = 1
    _resolution = 28
    _normalization_stats = {'mean': (0.1307,), 'std': (0.3081,)}
    _pil_mode = "L"

    def __init__(
        self,
        root,
        split,
        transform=None,
        resize=True,
        download=False,
        subsample_8=False,
        ontology='five-comp',
        augment=False,
    ):
        assert transform is None
        transform = get_transform_MNIST(resize=resize, augment=augment)
        self.subclass_proportions = {8: 0.05} if ('train' in split and subsample_8) else {}
        super().__init__(
            'MNIST', root, split, transform=transform, download=download, ontology=ontology
        )

    def _load_samples(self):
        """Loads the U-MNIST dataset from the data file created by self._download"""
        data_file = f'{self.split}.pt'
        logging.info(f'Loading {self.split} split...')
        data, original_labels = torch.load(os.path.join(self.processed_folder, data_file))

        logging.info('Original label counts:')
        logging.info(np.bincount(original_labels))

        # subsample some subset of subclasses
        if self.subclass_proportions:
            logging.info(f'Subsampling subclasses: {self.subclass_proportions}')
            data, original_labels = self.subsample_digits(
                data, original_labels, self.subclass_proportions
            )
            logging.info('New label counts:')
            logging.info(np.bincount(original_labels))

        # determine superclass partition of original_labels
        if self.ontology == 'five-comp':
            superclass_labels = (original_labels > 4).long()
            self.superclass_names = ['< 5', '≥ 5']
        else:
            raise ValueError(f'Ontology {self.ontology} not supported.')

        X = data
        Y_dict = {'superclass': superclass_labels, 'true_subclass': original_labels.clone()}
        return X, Y_dict

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (x_dict, y_dict) where x_dict is a dictionary mapping all
                possible inputs and y_dict is a dictionary for all possible labels.
        """
        x = self.X[idx]
        image = Image.fromarray(x.numpy(), mode=self._pil_mode)
        if self.transform is not None:
            image = self.transform(image)
        x = image

        y_dict = {k: v[idx] for k, v in self.Y_dict.items()}
        return x, y_dict

    def subsample_digits(self, data, labels, subclass_proportions, seed=0):
        prev_state = random.getstate()
        random.seed(seed)
        data_mod_seed = random.randint(0, 2 ** 32)
        random.seed(data_mod_seed)

        for label, freq in subclass_proportions.items():
            logging.info(f'Subsampling {label} fine class, keeping {freq*100} percent...')
            inds = [i for i, x in enumerate(labels) if x == label]
            inds = set(random.sample(inds, int((1 - freq) * len(inds))))
            labels = torch.tensor([lab for i, lab in enumerate(labels) if i not in inds])
            data = torch.stack([datum for i, datum in enumerate(data) if i not in inds])

        random.setstate(prev_state)
        return data, labels

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'MNIST', 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'MNIST', 'processed')

    def _check_exists(self):
        return all(
            os.path.exists(os.path.join(self.processed_folder, f'{split}.pt'))
            for split in ['train', 'val', 'test']
        )

    def _download(self):
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(
                url, download_root=self.raw_folder, filename=filename, md5=md5
            )

        # process and save as torch files
        logging.info('Processing...')
        training_set = (
            read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte')),
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte')),
        )
        with open(os.path.join(self.processed_folder, 'train.pt'), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, 'test.pt'), 'wb') as f:
            torch.save(test_set, f)
        logging.info('Done downloading!')

        self._create_val_split()

    def _create_val_split(self, seed=0, val_proportion=0.2):
        data, original_labels = torch.load(os.path.join(self.processed_folder, 'train.pt'))
        original_labels = original_labels.numpy()
        original_label_counts = np.bincount(original_labels)
        assert all(
            i > 0 for i in original_label_counts
        ), 'set(labels) must consist of consecutive numbers in [0, S]'
        val_quota = np.round(original_label_counts * val_proportion).astype(np.int)

        # reset seed here in case random fns called again (i.e. if get_loaders called twice)
        prev_state = random.getstate()
        random.seed(seed)
        shuffled_idxs = random.sample(range(len(data)), len(data))
        random.setstate(prev_state)

        train_idxs = []
        val_idxs = []
        val_counts = defaultdict(int)

        # Iterate through shuffled dataset to extract valset idxs
        for i in shuffled_idxs:
            label = original_labels[i]
            if val_counts[label] < val_quota[label]:
                val_idxs.append(i)
                val_counts[label] += 1
            else:
                train_idxs.append(i)

        train_idxs = sorted(train_idxs)
        val_idxs = sorted(val_idxs)
        assert (
            len(set(val_idxs) & set(train_idxs)) == 0
        ), 'valset and trainset must be mutually exclusive'

        logging.info(
            f'Creating training set with class counts:\n'
            + f'{np.bincount(original_labels[train_idxs])}'
        )
        trainset = (data[train_idxs], torch.tensor(original_labels[train_idxs]))
        with open(os.path.join(self.processed_folder, 'train.pt'), 'wb') as f:
            torch.save(trainset, f)

        logging.info(
            f'Creating validation set with class counts:\n'
            + f'{np.bincount(original_labels[val_idxs])}'
        )
        valset = (data[val_idxs], torch.tensor(original_labels[val_idxs]))
        with open(os.path.join(self.processed_folder, 'val.pt'), 'wb') as f:
            torch.save(valset, f)
        logging.info(f'Split complete!')


def read_label_file(path):
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert x.dtype == torch.uint8
    assert x.ndimension() == 1
    return x.long()


def read_image_file(path):
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert x.dtype == torch.uint8
    assert x.ndimension() == 3
    return x


def read_sn3_pascalvincent_tensor(path, strict=True):
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
    Argument may be a filename, compressed filename, or file object.
    """
    # typemap
    if not hasattr(read_sn3_pascalvincent_tensor, 'typemap'):
        read_sn3_pascalvincent_tensor.typemap = {
            8: (torch.uint8, np.uint8, np.uint8),
            9: (torch.int8, np.int8, np.int8),
            11: (torch.int16, np.dtype('>i2'), 'i2'),
            12: (torch.int32, np.dtype('>i4'), 'i4'),
            13: (torch.float32, np.dtype('>f4'), 'f4'),
            14: (torch.float64, np.dtype('>f8'), 'f8'),
        }
    # read
    with open_maybe_compressed_file(path) as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert nd >= 1 and nd <= 3
    assert ty >= 8 and ty <= 14
    m = read_sn3_pascalvincent_tensor.typemap[ty]
    s = [get_int(data[4 * (i + 1) : 4 * (i + 2)]) for i in range(nd)]
    parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
    assert parsed.shape[0] == np.prod(s) or not strict
    return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def open_maybe_compressed_file(path):
    """Return a file object that possibly decompresses 'path' on the fly.
    Decompression occurs when argument `path` is a string and ends with '.gz' or '.xz'.
    """
    if not isinstance(path, torch._six.string_classes):
        return path
    if path.endswith('.gz'):
        import gzip

        return gzip.open(path, 'rb')
    if path.endswith('.xz'):
        import lzma

        return lzma.open(path, 'rb')
    return open(path, 'rb')


def get_transform_MNIST(resize=True, augment=False):
    test_transform_list = [
        transforms.ToTensor(),
        transforms.Normalize(**MNISTDataset._normalization_stats),
    ]
    if resize:
        test_transform_list.insert(0, transforms.Resize((32, 32)))
    if not augment:
        return transforms.Compose(test_transform_list)

    train_transform_list = [
        transforms.RandomCrop(MNISTDataset._resolution, padding=4),
        transforms.RandomHorizontalFlip(),
    ] + test_transform_list
    return transforms.Compose(train_transform_list)
