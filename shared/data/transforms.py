from __future__ import annotations

from typing import ClassVar, Sequence

import torch
from torch.tensor import Tensor

from kornia.geometry import rotate

__all__ = ["NoisyDequantize", "Quantize", "Rotate"]


class Augmentation:
    """Base class for label-dependent augmentations."""

    def _augment(self, data: torch.Tensor) -> torch.Tensor:
        """Augment the input data

        Args:
            data: Tensor. Input data to be augmented.

        Returns:
            Tensor, augmented data
        """
        return data

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Calls the augment method on the the input data.

        Args:
            data: Tensor. Input data to be augmented.

        Returns:
            Tensor, augmented data
        """
        return self._augment(data)


class NoisyDequantize(Augmentation):
    def __init__(self, n_bits_x: int = 8):
        self.n_bins = 2 ** n_bits_x

    def _augment(self, data: torch.Tensor) -> torch.Tensor:
        return torch.clamp(data + (torch.rand_like(data) / self.n_bins), min=0, max=1)


class Quantize(Augmentation):
    def __init__(self, n_bits_x: int = 8):
        self.n_bits_x = n_bits_x
        self.n_bins = 2 ** n_bits_x

    def _augment(self, data: torch.Tensor) -> torch.Tensor:
        if self.n_bits_x < 8:
            # for n_bits_x=5, this turns the range (0, 1) to (0, 32) and floors it
            # the exact value 32 will only appear if there was an exact 1 in `data`
            x = torch.floor(torch.clamp(data, 0, 1 - 1e-6) * self.n_bins)
            # re-normalize to between 0 and 1
            x = x / self.n_bins
        return x


class Rotate(Augmentation):
    ...
    _pos_angles: ClassVar[Tensor] = torch.as_tensor([[0.0], [90.0], [180.0], [270.0]])
    _pos_labels: ClassVar[Tensor] = torch.as_tensor([[0], [1], [2], [3]])
    target_dim = 4

    def _augment(self, data: Tensor) -> Tensor:
        index = torch.randint(size=(data.size(0),), low=0, high=4)
        angles = self._pos_angles[index].squeeze(-1)
        return rotate(data, angles)
