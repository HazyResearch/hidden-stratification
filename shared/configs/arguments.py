from dataclasses import dataclass, field
import logging
import shlex
from typing import Dict, List, Type, TypeVar

import torch

from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig, MISSING
from shared.configs.enums import CelebaAttributes, IsicAttrs, QuantizationLevel

__all__ = [
    "BaseConfig",
    "BiasConfig",
    "CelebaConfig",
    "CmnistConfig",
    "DatasetConfig",
    "ImageDatasetConfig",
    "IsicConfig",
    "register_configs",
]


LOGGER = logging.getLogger(__name__.split(".")[-1].upper())


@dataclass
class DatasetConfig:
    """General data set settings."""

    _target_: str = "shared.configs.DatasetConfig"
    log_name: str = MISSING  # don't rely on this to check which dataset is loaded

    data_pcnt: float = 1.0  # data pcnt should be a real value > 0, and up to 1
    context_pcnt: float = 0.4
    test_pcnt: float = 0.2
    root: str = ""
    transductive: bool = False  # whether to include the test data in the pool of unlabelled data

    num_workers: int = 4
    data_split_seed: int = 42


@dataclass
class ImageDatasetConfig(DatasetConfig):
    """Settings specific to image datasets."""

    quant_level: QuantizationLevel = QuantizationLevel.eight  # number of bits that encode color
    input_noise: bool = False  # add uniform noise to the input


@dataclass
class CmnistConfig(ImageDatasetConfig):
    """Settings specific to the cMNIST dataset."""

    _target_: str = "shared.configs.CmnistConfig"
    log_name: str = "cmnist"

    # Colored MNIST settings
    scale: float = 0.0
    greyscale: bool = False
    background: bool = False
    black: bool = True
    binarize: bool = True
    rotate_data: bool = False
    shift_data: bool = False
    color_correlation: float = 1.0
    padding: int = 2  # by how many pixels to pad the cmnist images by
    filter_map_labels: Dict[str, int] = field(default_factory=dict)
    colors: List[int] = field(default_factory=list)


@dataclass
class CelebaConfig(ImageDatasetConfig):
    """Settings specific to the CelebA dataset."""

    _target_: str = "shared.configs.CelebaConfig"
    log_name: str = "celeba"

    # CelebA settings
    celeba_sens_attr: CelebaAttributes = CelebaAttributes.Male
    celeba_target_attr: CelebaAttributes = CelebaAttributes.Smiling


@dataclass
class IsicConfig(ImageDatasetConfig):
    """Settings specific to the ISIC dataset."""

    _target_: str = "shared.configs.IsicConfig"
    log_name: str = "isic"

    # ISIC settings
    isic_sens_attr: IsicAttrs = IsicAttrs.histo
    isic_target_attr: IsicAttrs = IsicAttrs.malignant


@dataclass
class BiasConfig:

    _target_: str = "shared.configs.BiasConfig"

    # Dataset manipulation
    missing_s: List[int] = field(default_factory=list)
    mixing_factor: float = 0  # How much of context should be mixed into training?
    adult_biased_train: bool = True  # if True, make the training set biased, based on mixing factor
    # the subsample flags work like this: you give it a class id and a fraction in the form of a
    # float. the class id is given by class_id = y * s_count + s, so for binary s and y, the
    # correspondance is like this:
    # 0: y=0/s=0, 1: y=0/s=1, 2: y=1/s=0, 3: y=1/s=1
    subsample_context: Dict[str, float] = field(default_factory=dict)
    subsample_train: Dict[str, float] = field(default_factory=dict)

    log_dataset: str = ""


T = TypeVar("T", bound="BaseConfig")


@dataclass
class BaseConfig:
    """Minimum config needed to do data loading."""

    _target_: str = "shared.configs.BaseConfig"
    cmd: str = ""

    data: DatasetConfig = MISSING
    bias: BiasConfig = MISSING
    lr: float = 1e-3
    epochs: int = 40

    @classmethod
    def from_hydra(cls: Type[T], hydra_config: DictConfig) -> T:
        """Instantiate this class based on a hydra config.

        This is necessary because dataclasses cannot be instantiated recursively yet.
        """
        subconfigs = {
            k: instantiate(v) for k, v in hydra_config.items() if k not in ("_target_", "cmd")
        }

        return cls(**subconfigs)


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(node=CmnistConfig, name="cmnist", package="data", group="data/schema")
    cs.store(node=CelebaConfig, name="celeba", package="data", group="data/schema")
    cs.store(node=IsicConfig, name="isic", package="data", group="data/schema")
    cs.store(node=BaseConfig, name="config_schema")
