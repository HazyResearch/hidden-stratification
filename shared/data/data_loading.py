import logging
import platform
from typing import Dict, NamedTuple, Optional, Tuple, Union

import ethicml as em
import ethicml.vision as emvi
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset, Subset
from torch.utils.data.dataset import ConcatDataset
from torchvision import transforms as TF
from torchvision.datasets import MNIST

from hydra.utils import to_absolute_path
from shared.configs import (
    BaseConfig,
    CelebaConfig,
    CmnistConfig,
    IsicConfig,
    QuantizationLevel,
)

from .dataset_wrappers import TensorDataTupleDataset
from .isic import IsicDataset
from .misc import shrink_dataset
from .transforms import NoisyDequantize, Quantize, Rotate

__all__ = ["DatasetTriplet", "load_dataset"]

LOGGER = logging.getLogger(__name__.split(".")[-1].upper())


class DatasetTriplet(NamedTuple):
    context: Dataset
    test: Dataset
    train: Dataset
    s_dim: int
    y_dim: int


class RawDataTuple(NamedTuple):
    x: Tensor
    s: Tensor
    y: Tensor


def load_dataset(cfg: BaseConfig) -> DatasetTriplet:
    context_data: Dataset
    test_data: Dataset
    train_data: Dataset
    args = cfg.data
    data_root = args.root or find_data_dir()

    # =============== get whole dataset ===================
    if isinstance(args, CmnistConfig):
        augs = []
        if args.padding > 0:
            augs.append(nn.ConstantPad2d(padding=args.padding, value=0))
        if args.quant_level is not QuantizationLevel.eight:
            augs.append(Quantize(args.quant_level.value))
        if args.input_noise:
            augs.append(NoisyDequantize(args.quant_level.value))
        if args.rotate_data:
            augs.append(Rotate())

        train_data = MNIST(root=data_root, download=True, train=True)
        test_data: Union[Tuple, Dataset]
        test_data = MNIST(root=data_root, download=True, train=False)

        num_classes = 10
        if args.filter_map_labels:
            num_classes = max(args.filter_map_labels.values()) + 1
            if any(i not in args.filter_map_labels.values() for i in range(num_classes)):
                raise ValueError("Some values are skipped in filter_map_labels.")

            def _filter_(dataset: MNIST):
                final_mask = torch.zeros_like(dataset.targets).bool()
                for old_label, new_label in args.filter_map_labels.items():
                    mask = dataset.targets == int(old_label)
                    dataset.targets[mask] = new_label
                    final_mask |= mask
                dataset.data = dataset.data[final_mask]
                dataset.targets = dataset.targets[final_mask]

            _filter_(train_data)
            _filter_(test_data)

        num_colors = len(args.colors) if len(args.colors) > 0 else num_classes
        colorizer = emvi.LdColorizer(
            scale=args.scale,
            background=args.background,
            black=args.black,
            binarize=args.binarize,
            greyscale=args.greyscale,
            color_indices=args.colors or None,
        )

        test_data = (test_data.data, test_data.targets)
        context_len = round(args.context_pcnt * len(train_data))
        train_len = len(train_data) - context_len
        split_sizes = (context_len, train_len)
        shuffle_inds = torch.randperm(len(train_data))
        context_data, train_data = zip(
            train_data.data[shuffle_inds].split(split_sizes),
            train_data.targets[shuffle_inds].split(split_sizes),
        )

        def _colorize_subset(
            _subset: Tuple[Tensor, Tensor],
            _apply_missing_s: bool,
            _correlation: float = 0.0,
        ) -> RawDataTuple:
            x, y = _subset
            x = x.unsqueeze(1).expand(-1, 3, -1, -1) / 255.0
            for aug in augs:
                x = aug(x)
            if _apply_missing_s and cfg.bias.missing_s:
                s_values = torch.tensor(
                    [i for i in range(num_colors) if i not in cfg.bias.missing_s]
                )
                indexes = torch.randint_like(y, low=0, high=s_values.size(0))
                s = s_values[indexes]
            else:
                s = y.clone()
                indexes = torch.rand(s.shape) > _correlation
                s[indexes] = torch.randint_like(s[indexes], low=0, high=num_colors)
            x_col = colorizer(x, s)
            return RawDataTuple(x=x_col, s=s, y=y)

        def _subsample_by_s_and_y(
            _data: RawDataTuple, _target_props: Dict[str, float]
        ) -> RawDataTuple:
            _x = _data.x
            _s = _data.s
            _y = _data.y
            smallest: Tuple[int, Optional[int], Optional[int]] = (int(1e10), None, None)
            for _class_id, _prop in _target_props.items():
                _class_id = int(_class_id)  # hydra doesn't allow ints as keys, so we have to cast
                assert 0 <= _prop <= 1, "proportions should be between 0 and 1"
                target_y = _class_id // num_classes
                target_s = _class_id % num_colors
                _indexes = (_y == int(target_y)) & (_s == int(target_s))
                _n_matches = len(_indexes.nonzero(as_tuple=False))
                if _n_matches == 0:
                    assert _prop == 0, f"no samples for this: {target_y}, {target_s} (_class_id)"
                    continue
                _num_to_keep = round(_prop * (_n_matches - 1))
                _to_keep = torch.randperm(_n_matches) < _num_to_keep
                _indexes[_indexes.nonzero(as_tuple=False)[_to_keep]] = False
                _x = _x[~_indexes]
                _s = _s[~_indexes]
                _y = _y[~_indexes]

                if _num_to_keep != 0 and _num_to_keep < smallest[0]:
                    smallest = (_num_to_keep, target_y, target_s)
            if smallest[1] is not None:
                LOGGER.info(
                    f"    Smallest cluster (y={smallest[1]}, s={smallest[2]}): {smallest[0]}"
                )
            return RawDataTuple(x=_x, s=_s, y=_y)

        # if missing_s is set, the following will remove those s values
        train_data_t = _colorize_subset(train_data, _apply_missing_s=True)
        if cfg.bias.subsample_train:
            if cfg.bias.missing_s:
                LOGGER.info("bias.missing_s & bias.subsample_train. hope ya know what you're doing")
            LOGGER.info("Subsampling training set...")
            train_data_t = _subsample_by_s_and_y(train_data_t, cfg.bias.subsample_train)
        test_data_t = _colorize_subset(test_data, _apply_missing_s=False)
        context_data_t = _colorize_subset(context_data, _apply_missing_s=False)

        if cfg.bias.subsample_context:
            LOGGER.info("Subsampling context set...")
            context_data_t = _subsample_by_s_and_y(context_data_t, cfg.bias.subsample_context)
            # test data remains balanced
            # test_data = _subsample_by_class(*test_data, args.subsample)

        train_data = TensorDataTupleDataset(train_data_t.x, train_data_t.s, train_data_t.y)
        test_data = TensorDataTupleDataset(test_data_t.x, test_data_t.s, test_data_t.y)
        context_data = TensorDataTupleDataset(context_data_t.x, context_data_t.s, context_data_t.y)

        y_dim = 1 if num_classes == 2 else num_classes
        s_dim = 1 if num_colors == 2 else num_colors

    elif isinstance(args, (CelebaConfig, IsicConfig)):
        if isinstance(args, CelebaConfig):
            tform_ls = [TF.Resize(64), TF.CenterCrop(64)]
        else:
            tform_ls = []
        tform_ls.append(TF.ToTensor())
        if args.quant_level is not QuantizationLevel.eight:
            tform_ls.append(Quantize(args.quant_level.value))
        if args.input_noise:
            tform_ls.append(NoisyDequantize(args.quant_level.value))
        tform_ls.append(TF.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = TF.Compose(tform_ls)

        y_dim = 1
        if isinstance(args, CelebaConfig):
            # unbiased_pcnt = args.test_pcnt + args.context_pcnt
            dataset, base_dir = em.celeba(
                download_dir=data_root,
                label=args.celeba_target_attr.name,
                sens_attr=args.celeba_sens_attr.name,
                download=True,
                check_integrity=True,
            )
            assert dataset is not None
            all_data = emvi.TorchImageDataset(
                data=dataset.load(), root=base_dir, transform=transform, target_transform=None
            )
            s_dim = all_data.s_dim
        else:
            all_data = IsicDataset(
                root=data_root,
                sens_attr=args.isic_sens_attr,
                target_attr=args.isic_target_attr,
                max_samples=400,
                download=True,
                transform=transform,
                target_transform=None,
            )
            s_dim = 1

        size = len(all_data)
        context_len = round(args.context_pcnt * size)
        test_len = round(args.test_pcnt * size)
        train_len = size - context_len - test_len

        context_inds, train_inds, test_inds = torch.randperm(size).split(
            (context_len, train_len, test_len)
        )

        def _subsample_inds_by_s_and_y(
            _data: Dataset, _subset_inds: Tensor, _target_props: Dict[str, float]
        ) -> Tensor:

            card_y = max(y_dim, 2)
            card_s = max(s_dim, 2)
            for _class_id, _prop in _target_props.items():
                _class_id = int(_class_id)  # hydra doesn't allow ints as keys, so we have to cast
                assert 0 <= _prop <= 1, "proportions should be between 0 and 1"
                _s = _data.s[_subset_inds]
                _y = _data.y[_subset_inds]
                target_y = _class_id // card_y
                target_s = _class_id % card_s
                _indexes = (_y == int(target_y)) & (_s == int(target_s))
                _n_matches = len(_indexes.nonzero(as_tuple=False))
                _to_keep = torch.randperm(_n_matches) < (round(_prop * (_n_matches - 1)))
                _indexes[_indexes.nonzero(as_tuple=False)[_to_keep]] = False
                _subset_inds = _subset_inds[~_indexes.squeeze()]

            return _subset_inds

        if cfg.bias.subsample_context:
            context_inds = _subsample_inds_by_s_and_y(
                all_data, context_inds, cfg.bias.subsample_context
            )
        if cfg.bias.subsample_train:
            train_inds = _subsample_inds_by_s_and_y(all_data, train_inds, cfg.bias.subsample_train)

        context_data = Subset(all_data, context_inds.tolist())
        train_data = Subset(all_data, train_inds.tolist())
        test_data = Subset(all_data, test_inds)
    else:
        raise ValueError(f"Invalid choice of dataset: {args}")

    if 0 < args.data_pcnt < 1:
        context_data = shrink_dataset(context_data, args.data_pcnt)
        train_data = shrink_dataset(train_data, args.data_pcnt)
        test_data = shrink_dataset(test_data, args.data_pcnt)
    # Enable transductive learning (i.e. using the test data for semi-supervised learning)
    if args.transductive:
        context_data = ConcatDataset([context_data, test_data])

    return DatasetTriplet(
        context=context_data,
        test=test_data,
        train=train_data,
        s_dim=s_dim,
        y_dim=y_dim,
    )


def find_data_dir() -> str:
    """Find data directory for the current machine based on predefined mappings."""
    data_dirs = {
        "fear": "/mnt/data0/data",
        "hydra": "/mnt/archive/shared/data",
        "m900382.inf.susx.ac.uk": "/Users/tk324/PycharmProjects/NoSINN/data",
        "turing": "/srv/galene0/shared/data",
    }
    name_of_machine = platform.node()  # name of machine as reported by operating system
    return data_dirs.get(name_of_machine, to_absolute_path("data"))
