"""Functions related to saving and loading results."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from shared.configs import BaseConfig

__all__ = ["ClusterResults", "save_results", "load_results"]

LOGGER = logging.getLogger(__name__.split(".")[-1].upper())


class ClusterResults(NamedTuple):
    """Information that the fcm code passes on to suds."""

    flags: Dict[str, Any]
    cluster_ids: torch.Tensor
    class_ids: torch.Tensor
    enc_path: Path
    context_metrics: Optional[Dict[str, float]]
    test_metrics: Optional[Dict[str, float]] = None


def save_results(save_path: Path, cluster_results: ClusterResults) -> Path:
    """Save a tensor in a file."""
    save_dict = {
        "args": cluster_results.flags,
        "cluster_ids": cluster_results.cluster_ids,
        "class_ids": cluster_results.class_ids,
        "enc_path": str(cluster_results.enc_path.resolve()),
        "context_metrics": cluster_results.context_metrics or {},
        "test_metrics": cluster_results.test_metrics or {},
    }
    torch.save(save_dict, save_path)
    LOGGER.info(
        f"To make use of the generated cluster labels:\n"
        f"misc.cluster_label_file={save_path.resolve()}"
    )
    return save_path


def load_results(cfg: BaseConfig, check: bool = True) -> ClusterResults:
    """Load a tensor from a file."""
    data = torch.load(cfg.misc.cluster_label_file, map_location=torch.device("cpu"))
    if check:
        saved_cfg = data["args"]
        assert (
            saved_cfg["data.log_name"] == cfg.data.log_name
        ), f'{saved_cfg["data.log_name"]} != {cfg.data.log_name}'
        assert (
            saved_cfg["data.data_pcnt"] == cfg.data.data_pcnt
        ), f'{saved_cfg["data.data_pcnt"]} != {cfg.data.data_pcnt}'
        assert (
            saved_cfg["misc.data_split_seed"] == cfg.misc.data_split_seed
        ), f'{saved_cfg["misc.data_split_seed"]} != {cfg.misc.data_split_seed}'
        assert (
            saved_cfg["data.context_pcnt"] == cfg.data.context_pcnt
        ), f'{saved_cfg["data.context_pcnt"]} != {cfg.data.context_pcnt}'
        assert (
            saved_cfg["data.test_pcnt"] == cfg.data.test_pcnt
        ), f'{saved_cfg["data.test_pcnt"]} != {cfg.data.test_pcnt}'
    class_ids = data["class_ids"] if "class_ids" in data else torch.zeros_like(data["cluster_ids"])
    return ClusterResults(
        flags=data["args"],
        cluster_ids=data["cluster_ids"],
        class_ids=class_ids,
        enc_path=Path(data.get("enc_path", "")),
        context_metrics=data.get("context_metrics", None),
        test_metrics=data.get("test_metrics", None),
    )
