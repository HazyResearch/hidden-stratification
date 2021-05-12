from typing import Dict, Tuple
from typing_extensions import Literal

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from fdm.models import Classifier
from shared.configs import BaseConfig
from shared.data.data_loading import DatasetTriplet

from .base import GEORGEDataset
from .classifier import fit_classifier
from .extraction import TrainContextWrapper, extract_labels_from_dataset

DATA_SPLITS = Literal['train', 'train_clean', 'val', 'test']


class FdmDatasetWrapper(GEORGEDataset):
    def __init__(self, cfg: BaseConfig, dataset_triplet: DatasetTriplet, split: DATA_SPLITS) -> None:
        self.cfg = cfg
        self.triplet = dataset_triplet
        super().__init__(name=cfg.data.log_name, root="no need to know", split=split)

    def _check_exists(self) -> bool:
        return True  # this always exists

    def _load_samples(self) -> Tuple[Dataset[Tuple[Tensor, Tensor, Tensor]], Dict[str, Tensor]]:
        if self.split in ("train", "train_clean", "val"):
            # 1. step: train classifier on triplet.train
            # 2. step: predict labels on triplet.context
            # 3. step: combine datasets
            dataset = train_and_predict(self.cfg, self.triplet)
        else:
            dataset = self.triplet.test
        # 4. step: extract labels
        s, y = extract_labels_from_dataset(dataset)
        y_dict = {"superclass": y, "true_subclass": s}
        return dataset, y_dict

    def __getitem__(self, idx: int):
        x, _, _ = self.X[idx]
        y_dict = {k: v[idx] for k, v in self.Y_dict.items()}
        return x, y_dict


def train_and_predict(cfg: BaseConfig, dataset_triplet: DatasetTriplet):
    train_data, test_data = dataset_triplet.train, dataset_triplet.context
    y_dim = dataset_triplet.y_dim
    input_shape = next(iter(train_data))[0].shape

    train_loader_kwargs = {}
    train_loader_kwargs["shuffle"] = True

    train_loader = DataLoader(
        train_data,
        batch_size=cfg.fdm.eval_batch_size,
        pin_memory=True,
        **train_loader_kwargs,
    )
    test_loader = DataLoader(
        test_data, batch_size=cfg.fdm.test_batch_size, shuffle=False, pin_memory=True
    )

    clf: Classifier = fit_classifier(
        cfg,
        input_shape,
        train_data=train_loader,
        train_on_recon=False,
        pred_s=False,
        test_data=test_loader,
        target_dim=y_dim,
        device=torch.device("gpu:0"),
    )

    preds, _, _ = clf.predict_dataset(test_loader, device=torch.device("gpu:0"))
    return TrainContextWrapper(train_data, context=test_data, new_context_labels=preds)
