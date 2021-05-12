import torch
from torch import Tensor
from typing import Union, Tuple
from ethicml.vision.data.image_dataset import TorchImageDataset
from torch.utils.data import ConcatDataset, Subset, Dataset

from shared.data.dataset_wrappers import DataTupleDataset, TensorDataTupleDataset
from shared.data.isic import IsicDataset

__all__ = ["TrainContextWrapper", "extract_labels_from_dataset"]


class TrainContextWrapper(Dataset):
    def __init__(self, train: Dataset, context: Dataset, new_context_labels: torch.Tensor):
        super().__init__()
        self.train = train
        self.context = context
        self.new_context_labels = new_context_labels
        self.train_length = len(train)
        self.length = self.train_length + len(context)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index) -> Tuple[Tensor, Tensor, Tensor]:
        if index < self.train_length:
            return self.train[index]
        else:
            x, s, _ = self.context[index - self.train_length]
            pred_y = self.new_context_labels[index - self.train_length]
            return x, s, pred_y


_Dataset = Union[
    TensorDataTupleDataset, DataTupleDataset, Subset[Union[TorchImageDataset, IsicDataset]]
]
ExtractableDataset = Union[ConcatDataset[_Dataset], _Dataset, TrainContextWrapper]


def extract_labels_from_dataset(dataset: ExtractableDataset) -> Tuple[Tensor, Tensor]:
    def _extract(dataset: _Dataset):
        if isinstance(dataset, Subset):
            _s = cast(Tensor, dataset.dataset.s[dataset.indices])  # type: ignore
            _y = cast(Tensor, dataset.dataset.y[dataset.indices])  # type: ignore
        elif isinstance(dataset, DataTupleDataset):
            _s = torch.as_tensor(dataset.s)
            _y = torch.as_tensor(dataset.y)
        else:
            _s = dataset.s
            _y = dataset.y
        return _s, _y

    if isinstance(dataset, ConcatDataset):
        s_all_ls, y_all_ls = [], []
        for _dataset in dataset.datasets:
            s, y = _extract(_dataset)  # type: ignore
            s_all_ls.append(s)
            y_all_ls.append(y)
        s_all = torch.cat(s_all_ls, dim=0)
        y_all = torch.cat(y_all_ls, dim=0)
    elif isinstance(dataset, TrainContextWrapper):
        s_all_ls, y_all_ls = [], []
        s, y = _extract(dataset.train)
        s_all_ls.append(s)
        y_all_ls.append(y)
        s, _ = _extract(dataset.context)
        s_all_ls.append(s)
        y_all_ls.append(dataset.new_context_labels)
        s_all = torch.cat(s_all_ls, dim=0)
        y_all = torch.cat(y_all_ls, dim=0)
    else:
        s_all, y_all = _extract(dataset)  # type: ignore
    return s_all, y_all
