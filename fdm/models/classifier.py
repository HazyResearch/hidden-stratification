from __future__ import annotations
import logging
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader, Dataset
from tqdm import trange

from ethicml.implementations.dro_modules import DROLoss
from fdm.models.base import ModelBase

__all__ = ["Classifier", "Regressor"]

LOGGER = logging.getLogger(__name__.split(".")[-1].upper())


class Classifier(ModelBase):
    """Wrapper for classifier models."""

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        optimizer_kwargs: dict[str, Any] | None = None,
        criterion: str | _Loss | None = None,
    ):
        """Build classifier model.

        Args:).
            n_classes: Positive integer. Number of class labels.
            model: nn.Module. Classifier model to wrap around.
            optimizer_args: Dictionary. Arguments to pass to the optimizer.

        Returns:
            None
        """
        super().__init__(model, optimizer_kwargs=optimizer_kwargs)
        if num_classes < 2:
            raise ValueError(
                f"Invalid number of classes: must equal 2 or more," f" {num_classes} given."
            )
        if criterion is None:
            if num_classes == 2:
                self.criterion = "bce"
            else:
                self.criterion = "ce"
        else:
            self.criterion = criterion
        self.num_classes = num_classes

    def apply_criterion(self, logits: Tensor, targets: Tensor) -> Tensor:
        if isinstance(self.criterion, str):
            if self.criterion == "bce":
                if targets.dtype != torch.float32:
                    targets = targets.float()
                logits = logits.view(-1, 1)
                targets = targets.view(-1, 1)
                return F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            elif self.criterion == "ce":
                targets = targets.view(-1)
                if targets.dtype != torch.long:
                    targets = targets.long()
                return F.cross_entropy(logits, targets, reduction="none")
            else:
                raise NotImplementedError("Only 'bce' and 'ce' losses are implemented using str.")
        elif isinstance(self.criterion, DROLoss):
            if isinstance(self.criterion.loss, nn.BCEWithLogitsLoss):
                if targets.dtype != torch.float32:
                    targets = targets.float()
                logits = logits.view(-1, 1)
                targets = targets.view(-1, 1)
                return self.criterion(logits, targets)
            elif isinstance(self.criterion.loss, nn.CrossEntropyLoss):
                targets = targets.view(-1)
                if targets.dtype != torch.long:
                    targets = targets.long()
                return self.criterion(logits, targets)
            else:
                raise NotImplementedError("Only 'bce' and 'ce' losses are implemented for DRO.")
        else:
            raise NotImplementedError("Only str and DROLoss are implemented.")

    def predict(self, inputs: Tensor, top: int = 1) -> Tensor:
        """Make prediction.

        Args:
            inputs: Tensor. Inputs to the classifier.
            top: Int. Top-k accuracy.

        Returns:
            Class predictions (tensor) for the given data samples.
        """
        outputs = super().__call__(inputs)
        if self.criterion == "bce" or (
            isinstance(self.criterion, DROLoss)
            and isinstance(self.criterion.loss, nn.BCEWithLogitsLoss)
        ):
            pred = torch.round(outputs.sigmoid())
        else:
            _, pred = outputs.topk(top, 1, True, True)

        return pred

    def predict_dataset(
        self, data: Dataset | DataLoader, device: torch.device, batch_size: int = 100
    ):
        if not isinstance(data, DataLoader):
            data = DataLoader(data, batch_size=batch_size, shuffle=False, pin_memory=True)
        preds, actual, sens = [], [], []
        with torch.set_grad_enabled(False):
            for x, s, y in data:
                x = x.to(device)
                y = y.to(device)

                batch_preds = self.predict(x)
                preds.append(batch_preds)
                actual.append(y)
                sens.append(s)

        preds = torch.cat(preds, dim=0).cpu().detach().view(-1)
        actual = torch.cat(actual, dim=0).cpu().detach().view(-1)
        sens = torch.cat(sens, dim=0).cpu().detach().view(-1)

        return preds, actual, sens

    def compute_accuracy(self, outputs: Tensor, targets: Tensor, top: int = 1) -> float:
        """Computes the classification accuracy.

        Args:
            outputs: Tensor. Classifier outputs.
            targets: Tensor. Targets for each input.
            top (int): Top-K accuracy.

        Returns:
            Accuracy of the predictions (float).
        """

        if self.criterion == "bce" or (
            isinstance(self.criterion, DROLoss)
            and isinstance(self.criterion.loss, nn.BCEWithLogitsLoss)
        ):
            pred = torch.round(outputs.sigmoid())
        else:
            _, pred = outputs.topk(top, 1, True, True)
        pred = pred.t().to(targets.dtype)
        correct = pred.eq(targets.view(1, -1).expand_as(pred)).float()
        correct = correct[:top].view(-1).float().sum(0, keepdim=True)
        accuracy = correct / targets.size(0) * 100

        return accuracy.detach().item()  # type: ignore

    def routine(
        self, data: Tensor, targets: Tensor, instance_weights: Tensor | None = None
    ) -> tuple[Tensor, float]:
        """Classifier routine.

        Args:
            data: Tensor. Input data to the classifier.
            targets: Tensor. Prediction targets.

        Returns:
            Tuple of classification loss (Tensor) and accuracy (float)
        """
        outputs = super().__call__(data)
        loss = self.apply_criterion(outputs, targets)
        if instance_weights is not None:
            loss = loss.view(-1) * instance_weights.view(-1)
        loss = loss.mean()
        acc = self.compute_accuracy(outputs, targets)

        return loss, acc

    def fit(
        self,
        train_data: Dataset | DataLoader,
        epochs: int,
        device: torch.device,
        test_data: Dataset | DataLoader | None = None,
        pred_s: bool = False,
        batch_size: int = 256,
        test_batch_size: int = 1000,
        **train_loader_kwargs: dict[str, Any],
    ):

        if not isinstance(train_data, DataLoader):
            # Default settings for train-loader
            train_loader_kwargs.setdefault("pin_memory", True)  # type: ignore
            train_loader_kwargs.setdefault("shuffle", True)  # type: ignore

            train_data = DataLoader(
                train_data,
                batch_size=batch_size,
                **train_loader_kwargs,
            )
        if test_data is not None:
            if not isinstance(test_data, DataLoader):
                test_data = DataLoader(
                    test_data,
                    batch_size=test_batch_size,
                    shuffle=False,
                    pin_memory=train_data.pin_memory,
                    num_workers=train_data.num_workers,
                )

        LOGGER.info("Training classifier...")
        pbar = trange(epochs)
        for epoch in pbar:
            self.model.train()

            for x, s, y in train_data:

                if pred_s:
                    target = s
                else:
                    target = y

                x = x.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                self.optimizer.zero_grad()
                loss, acc = self.routine(x, target)
                loss.backward()
                self.step()

            if test_data is not None:

                self.model.eval()
                sum_test_acc = 0.0
                # We could just obtain this count using len(dataloader.dataset) but then
                # the type-checker complains because a Dataset object doesn't have to implement
                # __len__ (it makes no sense for iterable datasets, for instance)
                num_samples = 0
                with torch.no_grad():
                    for x, s, y in test_data:

                        if pred_s:
                            target = s
                        else:
                            target = y

                        x = x.to(device)
                        target = target.to(device)

                        loss, acc = self.routine(x, target)
                        sum_test_acc += acc * target.size(0)  # undo the batch-wise averaging
                        num_samples += target.size(0)
                avg_test_acc = sum_test_acc / num_samples
                pbar.set_postfix(epoch=epoch + 1, avg_test_acc=avg_test_acc)
            else:
                pbar.set_postfix(epoch=epoch + 1)

        pbar.close()


class Regressor(Classifier):
    """Wrapper for regression models."""

    def __init__(self, model, optimizer_kwargs: dict[str, Any] | None = None):
        """Build classifier model.

        Args:
            model: nn.Module. Classifier model to wrap around.
            optimizer_args: Dictionary. Arguments to pass to the optimizer.

        Returns:
            None
        """
        super().__init__(model, 2, optimizer_kwargs=optimizer_kwargs)
        self.criterion = "mse"

    def apply_criterion(self, logits: Tensor, targets: Tensor) -> Tensor:
        return F.mse_loss(logits, targets.flatten(start_dim=1), reduction="none")

    def predict(self, inputs: Tensor, top: int = 1) -> Tensor:
        """Make prediction."""
        return super().__call__(inputs)

    def compute_accuracy(self, outputs: Tensor, targets: Tensor, top: int = 1) -> float:
        return 0
