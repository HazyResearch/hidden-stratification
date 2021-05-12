from __future__ import annotations
from collections.abc import Iterator, Sequence
from typing import List, Tuple

import torch
from torch import Tensor
from torch.utils.data import Sampler

__all__ = ["StratifiedSampler"]


class StratifiedSampler(Sampler[int]):
    r"""Samples equal proportion of elements from ``[0,..,len(group_ids)-1]``.

    To drop certain groups, set their multiplier to 0.

    Args:
        group_ids: a sequence of group IDs, not necessarily contiguous.
        num_samples_per_group: number of samples to draw per group. Note that if a multiplier is > 1
            then effectively more samples will be drawn for that group.
        replacement: if ``True``, samples are drawn with replacement. If not, they are drawn without
            replacement, which means that when a sample index is drawn for a row, it cannot be drawn
            again for that row.
        multiplier: an optional dictionary that maps group IDs to multipliers. If a multiplier is
            greater than 1, the corresponding group will be sampled at twice the rate as the other
            groups. If a multiplier is 0, the group will be skipped.

    Example:
        >>> list(StratifiedSampler([0, 0, 0, 0, 1, 1, 2], 10, replacement=True))
        [3, 5, 6, 3, 5, 6, 0, 5, 6]
        >>> list(StratifiedSampler([0, 0, 0, 0, 1, 1, 2], 10, replacement=True, multiplier={2: 2}))
        [3, 4, 6, 6, 3, 5, 6, 6, 1, 5, 6, 6]
        >>> list(StratifiedSampler([0, 0, 0, 0, 1, 1, 1, 2, 2], 7, replacement=False))
        [2, 6, 7, 0, 5, 8]
    """
    groupwise_idx: List[Tuple[Tensor, int]]  # use typing.Tuple here, because pytorch evals this
    num_groups_effective: int
    num_samples_per_group: int
    replacement: bool

    def __init__(
        self,
        group_ids: Sequence[int],
        num_samples_per_group: int,
        replacement: bool = True,
        multipliers: dict[int, int] | None = None,
    ):
        if (
            not isinstance(num_samples_per_group, int)
            or isinstance(num_samples_per_group, bool)
            or num_samples_per_group <= 0
        ):
            raise ValueError(
                f"num_samples_per_group should be a positive integer; got {num_samples_per_group}"
            )
        if not isinstance(replacement, bool):
            raise ValueError(
                f"replacement should be a boolean value, but got replacement={replacement}"
            )
        self.num_samples_per_group = num_samples_per_group
        self.replacement = replacement
        multipliers_ = {} if multipliers is None else multipliers

        group_ids_t = torch.as_tensor(group_ids, dtype=torch.int64)
        # find all unique IDs
        groups: list[int] = group_ids_t.unique().tolist()

        # get the indexes for each group separately and compute the effective number of groups
        groupwise_idx: list[tuple[Tensor, int]] = []
        num_groups_effective = 0
        for group in groups:
            idx = (group_ids_t == group).nonzero(as_tuple=False)
            multiplier = multipliers_.get(group, 1)
            assert isinstance(multiplier, int) and multiplier >= 0, "multiplier has to be >= 0"
            groupwise_idx.append((idx, multiplier))
            num_groups_effective += multiplier

            if not replacement and len(idx) < num_samples_per_group * multiplier:
                raise ValueError(
                    f"Not enough samples in group {group} to sample {num_samples_per_group}."
                )

        self.groupwise_idx = groupwise_idx
        self.num_groups_effective = num_groups_effective

    def __iter__(self) -> Iterator[int]:
        sampled_idx = []
        # loop over the groups and sample from each group separately
        for group_idx, multiplier in self.groupwise_idx:
            if self.replacement:
                for _ in range(multiplier):
                    # sampling with replacement:
                    # just sample enough random numbers to fill the quota
                    idx_of_idx = torch.randint(
                        low=0, high=len(group_idx), size=(self.num_samples_per_group,)
                    )
                    sampled_idx.append(group_idx[idx_of_idx])
            else:
                # sampling without replacement:
                # first shuffle the indexes and then take as many as we need
                shuffled_idx = group_idx[torch.randperm(len(group_idx))]
                # all elements in `sampled_idx` have to have the same size,
                # so we split the tensor in equal-sized parts and then take as many as we need
                chunks = torch.split(shuffled_idx, self.num_samples_per_group)
                sampled_idx += list(chunks[:multiplier])

        # interleave the sampled indexes so that they always appear in the same proportion
        interleaved_sampled_idx = torch.stack(sampled_idx, dim=1).view(-1)
        return iter(interleaved_sampled_idx.tolist())

    def __len__(self) -> int:
        return self.num_samples_per_group * self.num_groups_effective
