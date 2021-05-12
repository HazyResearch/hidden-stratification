from __future__ import annotations
from collections.abc import Mapping
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import ethicml as em
import pandas as pd
import wandb

from .utils import wandb_log

if TYPE_CHECKING:
    from shared.configs import BaseConfig


__all__ = ["compute_metrics", "make_tuple_from_data", "print_metrics", "write_results_to_csv"]

log = logging.getLogger(__name__.split(".")[-1].upper())


def make_tuple_from_data(
    train: em.DataTuple, test: em.DataTuple, pred_s: bool
) -> tuple[em.DataTuple, em.DataTuple]:
    train_x = train.x
    test_x = test.x

    if pred_s:
        train_y = train.s
        test_y = test.s
    else:
        train_y = train.y
        test_y = test.y

    return em.DataTuple(x=train_x, s=train.s, y=train_y), em.DataTuple(x=test_x, s=test.s, y=test_y)


def compute_metrics(
    cfg: BaseConfig,
    predictions: em.Prediction,
    actual: em.DataTuple,
    model_name: str,
    step: int,
    s_dim: int,
    exp_name: str | None = None,
    save_summary: bool = False,
    use_wandb: bool = False,
    additional_entries: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Compute accuracy and fairness metrics and log them.

    Args:
        args: args object
        predictions: predictions in a format that is compatible with EthicML
        actual: labels for the predictions
        model_name: name of the model used
        step: step of training (needed for logging to W&B)
        s_dim: dimension of s
        exp_name: name of the experiment
        save_summary: if True, a summary will be saved to wandb
        use_wandb: whether to use wandb at all
        additional_entries: entries that should go with in the summary
    Returns:
        dictionary with the computed metrics
    """

    predictions._info = {}
    metrics = em.run_metrics(
        predictions,
        actual,
        metrics=[em.Accuracy(), em.TPR(), em.TNR(), em.RenyiCorrelation()],
        per_sens_metrics=[em.Accuracy(), em.ProbPos(), em.TPR(), em.TNR()],
        diffs_and_ratios=s_dim < 4,  # this just gets too much with higher s dim
    )
    # replace the slash; it's causing problems
    metrics = {k.replace("/", "÷"): v for k, v in metrics.items()}
    metrics = {f"{k} ({model_name})": v for k, v in metrics.items()}
    if exp_name:
        metrics = {f"{exp_name}/{k}": v for k, v in metrics.items()}

    if use_wandb:
        wandb_log(cfg.misc, metrics, step=step)

        if save_summary:
            external = additional_entries or {}

            for metric_name, value in metrics.items():
                wandb.run.summary[metric_name] = value
            for metric_name, value in external.items():
                wandb.run.summary[metric_name] = value

    log.info(f"Results for {exp_name or ''} ({model_name}):")
    print_metrics(metrics)
    return metrics


def print_metrics(metrics: Mapping[str, int | float | str]) -> None:
    """Print metrics such that they don't clutter everything too much."""
    for key, value in metrics.items():
        log.info(f"    {key}: {value:.3g}")
    log.info("---")


def write_results_to_csv(results: Mapping[str, int | float | str], csv_dir: Path, csv_file: str):
    to_log = {}
    # to_log.update(flatten_dict(as_pretty_dict(cfg)))
    to_log.update(results)
    # I don't know why it has to be done in 2 steps, but it that's how it is
    results_df = pd.DataFrame(columns=list(to_log))
    results_df = results_df.append(to_log, ignore_index=True, sort=False)

    csv_dir.mkdir(exist_ok=True, parents=True)

    results_path = csv_dir / csv_file
    if results_path.exists():
        # load previous results and append new results
        previous_results = pd.read_csv(results_path)
        results_df = pd.concat(
            [previous_results, results_df], sort=False, ignore_index=True, axis="index"
        )
    results_df.reset_index(drop=True).to_csv(results_path, index=False)
    log.info(f"Results have been written to {results_path.resolve()}")
