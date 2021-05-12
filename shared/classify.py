from __future__ import annotations
from typing import Sequence

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet

from shared.configs import BaseConfig, CelebaConfig, CmnistConfig, ImageDatasetConfig
from shared.configs.classifiers import FcNet, Mp32x23Net, Mp64x64Net
from shared.models import Classifier
from shared.utils import ModelFn, compute_metrics, make_tuple_from_data, prod


def fit_classifier(
    data_config: BaseConfig,
    input_shape: Sequence[int],
    train_data: DataLoader,
    target_dim: int,
    device: torch.device,
    lr: float,
    epochs: int,
) -> Classifier:
    input_dim = input_shape[0]
    optimizer_kwargs = {"lr": lr}
    clf_fn: ModelFn

    if isinstance(data_config.data, ImageDatasetConfig):
        if isinstance(data_config.data, CmnistConfig):
            clf_fn = Mp32x23Net(batch_norm=True)
        elif isinstance(data_config.data, CelebaConfig):
            clf_fn = Mp64x64Net(batch_norm=True)
        else:  # ISIC dataset

            def resnet50_ft(input_dim: int, target_dim: int) -> ResNet:
                classifier = resnet50(pretrained=True)
                classifier.fc = nn.Linear(classifier.fc.in_features, target_dim)
                return classifier

            clf_fn = resnet50_ft
    else:

        def _adult_fc_net(input_dim: int, target_dim: int) -> nn.Sequential:
            encoder = FcNet(hidden_dims=[35])(input_dim=input_dim, target_dim=35)
            classifier = nn.Linear(35, target_dim)
            return nn.Sequential(encoder, classifier)

        optimizer_kwargs["weight_decay"] = 1e-8
        clf_fn = _adult_fc_net
    clf_base = clf_fn(input_dim, target_dim=target_dim)

    num_classes = max(2, target_dim)
    clf: Classifier = Classifier(
        clf_base, num_classes=num_classes, optimizer_kwargs=optimizer_kwargs
    )
    clf.to(device)
    clf.fit(
        train_data,
        test_data=None,
        epochs=epochs,
        device=device,
        pred_s=False,
    )

    return clf
