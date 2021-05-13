from __future__ import annotations

from torch import Tensor
import torch.nn as nn

__all__ = ["Mp64x64Net"]


class Mp64x64Net(nn.Module):
    def __init__(self, num_classes: int, batch_norm: bool = True):
        super().__init__()
        self.batch_norm = batch_norm

        layers = []
        layers.extend(self._conv_block(3, 64, 5, 1, 0))
        layers += [nn.MaxPool2d(2, 2)]

        layers.extend(self._conv_block(64, 128, 3, 1, 1))
        layers += [nn.MaxPool2d(2, 2)]

        layers.extend(self._conv_block(128, 128, 3, 1, 1))
        layers += [nn.MaxPool2d(2, 2)]

        layers.extend(self._conv_block(128, 256, 3, 1, 1))
        layers += [nn.MaxPool2d(2, 2)]

        layers.extend(self._conv_block(256, 512, 3, 1, 1))
        layers += [nn.MaxPool2d(2, 2)]

        layers += [nn.Flatten()]
        layers += [nn.Linear(512, num_classes)]

        self.seq = nn.Sequential(*layers)

    def _conv_block(
        self, in_dim: int, out_dim: int, kernel_size: int, stride: int, padding: int
    ) -> list[nn.Module]:
        _block: list[nn.Module] = []
        _block += [
            nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        ]
        if self.batch_norm:
            _block += [nn.BatchNorm2d(out_dim)]
        _block += [nn.LeakyReLU()]
        return _block

    def forward(self, x: Tensor) -> Tensor:
        return self.seq(x)
