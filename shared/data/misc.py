import csv
import os
from itertools import groupby
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler, Subset, random_split
from torch.utils.data._utils.collate import default_collate_err_msg_format, np_str_obj_array_pattern

__all__ = [
    "train_test_split",
    "shrink_dataset",
    "RandomSampler",
    "group_features",
    "set_transform",
    "grouped_features_indexes",
    "adaptive_collate",
]


def train_test_split(dataset: Dataset, train_pcnt: float) -> List[Subset]:
    curr_len = len(dataset)
    train_len = round(train_pcnt * curr_len)
    test_len = curr_len - train_len

    return random_split(dataset, lengths=[train_len, test_len])


def shrink_dataset(dataset: Dataset, pcnt: float) -> Subset:
    return train_test_split(dataset, train_pcnt=pcnt)[0]


def set_transform(dataset, transform):
    if hasattr(dataset, "dataset"):
        set_transform(dataset.dataset, transform)
    elif isinstance(dataset, Dataset):
        if hasattr(dataset, "transform"):
            dataset.transform = transform


class RandomSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.
    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
    """

    def __init__(self, data_source, replacement=False, num_samples=None):
        super(RandomSampler, self).__init__(data_source)
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples

        if not isinstance(self.replacement, bool):
            raise ValueError(
                "replacement should be a boolean value, but got "
                "replacement={}".format(self.replacement)
            )

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())

        return iter(torch.randperm(n)[: self.num_samples].tolist())

    def __len__(self):
        return self.num_samples


def data_tuple_to_dataset_sample(data, sens, target, root: str, filename: str) -> None:
    """

    Args:
        root: String. Root directory in which to save the dataset.
        filename: String. Filename of the sample being saved

    Returns:

    """
    if data.size(0) == 1:
        data = data.squeeze(0)
    # Create the root directory if it doesn't already exist
    if not os.path.exists(root):
        os.mkdir(root)
    # save the image
    if not filename.lower().endswith(".npz"):
        filename += ".npz"
    np.savez_compressed(os.path.join(root, filename), img=data.cpu().detach().numpy())
    # save filenames
    with open(os.path.join(root, "filename.csv"), "a", newline="") as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerow([filename])
    # save sensitive/nuisance labels
    with open(os.path.join(root, "sens.csv"), "ab") as f:
        sens = sens.view(-1)
        np.savetxt(f, sens.cpu().detach().numpy(), delimiter=",")
    # save targets
    with open(os.path.join(root, "target.csv"), "ab") as f:
        target = target.view(-1)
        np.savetxt(f, target.cpu().detach().numpy(), delimiter=",")


def group_features(disc_feats: List[str]) -> Iterator[Tuple[str, Iterator[str]]]:
    """Group discrete features names according to the first segment of their name"""

    def _first_segment(feature_name: str) -> str:
        return feature_name.split("_")[0]

    return groupby(disc_feats, _first_segment)


def grouped_features_indexes(disc_feature_groups: Dict[str, List[str]]) -> List[slice]:
    """Group discrete features names according to the first segment of their name
    and return a list of their corresponding slices (assumes order is maintained).
    """
    feature_slices = []
    start_idx = 0
    for group in disc_feature_groups.values():
        len_group = len(list(group))
        indexes = slice(start_idx, start_idx + len_group)
        feature_slices.append(indexes)
        start_idx += len_group

    return feature_slices


def adaptive_collate(batch: Any) -> Any:
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        ndims = elem.dim()
        if ndims > 0 and ndims % 2 == 0:
            return torch.cat(batch, dim=0, out=out)
        else:
            return torch.stack(batch, dim=0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        elem = batch[0]
        if elem_type.__name__ == "ndarray":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))
            return adaptive_collate([torch.as_tensor(b) for b in batch])
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(adaptive_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, (tuple, list)):
        transposed = zip(*batch)
        return [adaptive_collate(samples) for samples in transposed]
    raise TypeError(default_collate_err_msg_format.format(elem_type))
