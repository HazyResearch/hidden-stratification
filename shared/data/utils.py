from typing import NamedTuple

from torch import Tensor

__all__ = ["Batch"]


class Batch(NamedTuple):
    """A data structure for reducing clutter."""

    x: Tensor
    s: Tensor
    y: Tensor
