from typing import Sequence, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50

from fdm.models import Classifier
from shared.configs import Mp32x23Net, Mp64x64Net, FcNet
from shared.configs import BaseConfig, ImageDatasetConfig, CmnistConfig, CelebaConfig
from shared.utils import ModelFn, prod


def fit_classifier(
    cfg: BaseConfig,
    input_shape: Sequence[int],
    train_data: DataLoader,
    train_on_recon: bool,
    pred_s: bool,
    target_dim: int,
    device: torch.device,
    test_data: Optional[DataLoader] = None,
) -> Classifier:
    input_dim = input_shape[0]
    optimizer_kwargs = {"lr": cfg.lr}
    clf_fn: ModelFn

    if train_on_recon:
        if isinstance(cfg.data, ImageDatasetConfig):
            if isinstance(cfg.data, CmnistConfig):
                clf_fn = Mp32x23Net(batch_norm=True)
            elif isinstance(cfg.data, CelebaConfig):
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
    else:
        clf_fn = FcNet(hidden_dims=None)
        input_dim = prod(input_shape)
    clf_base = clf_fn(input_dim, target_dim=target_dim)

    num_classes = max(2, target_dim)
    clf: Classifier = Classifier(
        clf_base, num_classes=num_classes, optimizer_kwargs=optimizer_kwargs
    )
    clf.to(torch.device(device))
    clf.fit(
        train_data,
        test_data=test_data,
        epochs=cfg.epochs,
        device=torch.device(device),
        pred_s=pred_s,
    )

    return clf
