import logging

import torch
import numpy as np


class LossComputer:
    def __init__(self, criterion, is_robust, n_groups, group_counts, robust_step_size, stable=True,
                 size_adjustments=None, groups_outside_max=set(),
                 groups_outside_max_weight_factor=1., robust_then_average=True, use_cuda=True):
        self.criterion = criterion
        self.is_robust = is_robust
        self.n_groups = n_groups
        self.group_range = torch.arange(self.n_groups).unsqueeze(1).long()
        if use_cuda: self.group_range = self.group_range.cuda()

        if self.is_robust:
            self.robust_step_size = robust_step_size
            logging.info(f'Using robust loss with inner step size {self.robust_step_size}')
            self.stable = stable
            self.group_counts = group_counts.to(self.group_range.device)
            self.groups_outside_max = groups_outside_max
            self.groups_outside_max_tensor = torch.tensor([
                i in self.groups_outside_max for i in range(self.n_groups)
            ]).to(self.group_range.device)
            self.groups_in_max_tensor = torch.logical_not(self.groups_outside_max_tensor)
            self.n_groups_in_max = self.n_groups - len(self.groups_outside_max)
            self.erm_gfac = groups_outside_max_weight_factor / (1 +
                                                                groups_outside_max_weight_factor)
            self.robust_then_average = robust_then_average

            if size_adjustments is not None:
                self.adj = torch.tensor(size_adjustments).float().to(self.group_range.device)
                self.do_adj = True
            else:
                self.adj = torch.zeros(self.n_groups_in_max).float().to(self.group_range.device)
                self.do_adj = False
            self.loss_adjustment = self.adj / torch.sqrt(
                self.group_counts[self.groups_in_max_tensor])

            logging.info(
                f'Groups in maximization: {list(set(range(n_groups)) - self.groups_outside_max)}')
            logging.info(f'Groups outside maximization: {list(self.groups_outside_max)}')
            logging.info(
                f'Per-group loss adjustments: {np.round(self.loss_adjustment.tolist(), 2)}')
            # The following quantities are maintained/updated throughout training
            if self.stable:
                logging.info('Using numerically stabilized DRO algorithm')
                self.adv_probs_logits = torch.zeros(self.n_groups_in_max).to(
                    self.group_range.device)
            else:  # for debugging purposes
                logging.warn('Using original DRO algorithm')
                self.adv_probs = torch.ones(self.n_groups_in_max).to(
                    self.group_range.device) / self.n_groups_in_max
        else:
            logging.info('Using ERM')

    def loss(self, yhat, y, group_idx=None, is_training=False):
        # compute per-sample and per-group losses
        per_sample_losses = self.criterion(yhat, y)
        batch_size = y.shape[0]

        group_losses, group_counts = self.compute_group_avg(per_sample_losses, group_idx)
        corrects = (torch.argmax(yhat, 1) == y).float()
        group_accs, group_counts = self.compute_group_avg(corrects, group_idx)

        # compute overall loss
        if self.is_robust:
            robust_group_losses = group_losses[self.groups_in_max_tensor]
            nonrobust_group_losses = group_losses[self.groups_outside_max_tensor]
            nonrobust_group_counts = group_counts[self.groups_outside_max_tensor]
            nonrobust_group_fracs = nonrobust_group_counts / batch_size
            nonrobust_loss = nonrobust_group_losses @ nonrobust_group_fracs
            if self.robust_then_average:
                # Compute robust loss over groups in the maximization
                loss_a, _ = self.compute_robust_loss(robust_group_losses)
                # Compute sample average loss over the remaining groups
                loss_b = nonrobust_loss
                # Add them together with the correct proportions
                loss_a = loss_a * (1 - torch.sum(nonrobust_group_fracs))
                loss = 2 * ((1 - self.erm_gfac) * loss_a + self.erm_gfac * loss_b)
            else:
                robust_group_counts = group_counts[self.groups_in_max_tensor]
                robust_group_loss_totals = robust_group_losses * robust_group_counts
                nonrobust_count = torch.sum(nonrobust_group_counts)
                nonrobust_loss_total = nonrobust_group_losses @ nonrobust_group_counts
                # Add the "non-robust loss" to each group loss in the maximization
                # and average appropriately
                numerator = robust_group_loss_totals + nonrobust_loss_total
                denom = robust_group_counts + nonrobust_count
                denom = denom + (denom == 0).float()
                robust_group_losses = numerator / denom
                loss, _ = self.compute_robust_loss(robust_group_losses)
        else:
            loss = per_sample_losses.mean()

        return loss, (per_sample_losses, corrects), (group_losses, group_accs, group_counts)

    def compute_robust_loss(self, group_loss):
        if torch.is_grad_enabled():  # update adv_probs if in training mode
            adjusted_loss = group_loss
            if self.do_adj:
                adjusted_loss += self.loss_adjustment
            logit_step = self.robust_step_size * adjusted_loss.data
            if self.stable:
                self.adv_probs_logits = self.adv_probs_logits + logit_step
            else:
                self.adv_probs = self.adv_probs * torch.exp(logit_step)
                self.adv_probs = self.adv_probs / self.adv_probs.sum()

        if self.stable:
            adv_probs = torch.softmax(self.adv_probs_logits, dim=-1)
        else:
            adv_probs = self.adv_probs
        robust_loss = group_loss @ adv_probs
        return robust_loss, adv_probs

    def compute_group_avg(self, losses, group_idx, num_groups=None):
        # compute observed counts and mean loss for each group
        if num_groups is None:
            group_range = self.group_range
        else:
            group_range = torch.arange(num_groups).unsqueeze(1).long().to(group_idx.device)
        group_map = (group_idx == group_range).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count == 0).float()  # avoid nans
        group_loss = (group_map @ losses.view(-1)) / group_denom
        return group_loss, group_count

    def __call__(self, yhat, y, group_idx):
        return self.loss(yhat, y, group_idx)


def init_criterion(criterion_config, robust, trainset, use_cuda):
    num_subclasses = trainset.get_num_classes('subclass')
    subclass_counts = trainset.get_class_counts('subclass')

    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    if robust:
        groups_outside_max = criterion_config.get('groups_outside_max', set())
        size_adjustments = [criterion_config.get('size_adjustment', 0)] * \
                           (num_subclasses - len(groups_outside_max))
    else:
        groups_outside_max = set()
        size_adjustments = None
    fac = criterion_config.get('groups_outside_max_weight_factor', 1)
    criterion = LossComputer(criterion, robust, num_subclasses, subclass_counts,
                             criterion_config['robust_lr'], stable=criterion_config['stable_dro'],
                             size_adjustments=size_adjustments,
                             groups_outside_max=groups_outside_max,
                             groups_outside_max_weight_factor=fac,
                             robust_then_average=criterion_config.get('robust_then_average',
                                                                      True), use_cuda=use_cuda)
    return criterion
