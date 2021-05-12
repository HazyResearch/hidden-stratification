import logging

import numpy as np
import torch


class LossComputer:
    def __init__(
        self,
        criterion,
        is_robust,
        n_groups,
        group_counts,
        robust_step_size,
        stable=True,
        size_adjustments=None,
        auroc_version=False,
        class_map=None,
        use_cuda=True,
    ):
        self.criterion = criterion
        self.is_robust = is_robust
        self.auroc_version = auroc_version
        self.n_groups = n_groups
        if auroc_version:
            assert class_map is not None
            self.n_gdro_groups = len(class_map[0]) * len(class_map[1])
            self.class_map = class_map
        else:
            self.n_gdro_groups = n_groups
        self.group_range = torch.arange(self.n_groups).unsqueeze(1).long()
        if use_cuda:
            self.group_range = self.group_range.cuda()

        if self.is_robust:
            self.robust_step_size = robust_step_size
            logging.info(f'Using robust loss with inner step size {self.robust_step_size}')
            self.stable = stable
            self.group_counts = group_counts.to(self.group_range.device)

            if size_adjustments is not None:
                self.do_adj = True
                if auroc_version:
                    self.adj = torch.tensor(size_adjustments[0]).float().to(self.group_range.device)
                    self.loss_adjustment = self.adj / torch.sqrt(self.group_counts[:-1])
                else:
                    self.adj = torch.tensor(size_adjustments).float().to(self.group_range.device)
                    self.loss_adjustment = self.adj / torch.sqrt(self.group_counts)
            else:
                self.adj = torch.zeros(self.n_gdro_groups).float().to(self.group_range.device)
                self.do_adj = False
                self.loss_adjustment = self.adj

            logging.info(
                f'Per-group loss adjustments: {np.round(self.loss_adjustment.tolist(), 2)}'
            )
            # The following quantities are maintained/updated throughout training
            if self.stable:
                logging.info('Using numerically stabilized DRO algorithm')
                self.adv_probs_logits = torch.zeros(self.n_gdro_groups).to(self.group_range.device)
            else:  # for debugging purposes
                logging.warn('Using original DRO algorithm')
                self.adv_probs = (
                    torch.ones(self.n_gdro_groups).to(self.group_range.device) / self.n_gdro_groups
                )
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
            if self.auroc_version:
                neg_subclasses, pos_subclasses = self.class_map[0], self.class_map[1]
                pair_losses = []
                for neg_subclass in neg_subclasses:
                    neg_count = group_counts[neg_subclass]
                    neg_sbc_loss = group_losses[neg_subclass] * neg_count
                    for pos_subclass in pos_subclasses:
                        pos_count = group_counts[pos_subclass]
                        pos_sbc_loss = group_losses[pos_subclass] * pos_count
                        tot_count = neg_count + pos_count
                        tot_count = tot_count + (tot_count == 0).float()
                        pair_loss = (neg_sbc_loss + pos_sbc_loss) / tot_count
                        pair_losses.append(pair_loss)
                loss, _ = self.compute_robust_loss(torch.cat([l.view(1) for l in pair_losses]))
            else:
                loss, _ = self.compute_robust_loss(group_losses)
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

    def compute_group_avg(self, losses, group_idx, num_groups=None, reweight=None):
        # compute observed counts and mean loss for each group
        if num_groups is None:
            group_range = self.group_range
        else:
            group_range = torch.arange(num_groups).unsqueeze(1).long().to(group_idx.device)

        if reweight is not None:
            group_loss, group_count = [], []
            reweighted = losses * reweight
            for i in range(num_groups):
                inds = group_idx == i
                group_losses = reweighted[inds]
                group_denom = torch.sum(reweight[inds])
                group_denom = group_denom
                group_loss.append(
                    torch.sum(group_losses) / (group_denom + (group_denom == 0).float())
                )
                group_count.append(group_denom)
            group_loss, group_count = torch.tensor(group_loss), torch.tensor(group_count)
        else:
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
        size_adjustments = [criterion_config.get('size_adjustment', 0)] * num_subclasses
    else:
        size_adjustments = None
    criterion = LossComputer(
        criterion,
        robust,
        num_subclasses,
        subclass_counts,
        criterion_config['robust_lr'],
        stable=criterion_config['stable_dro'],
        size_adjustments=size_adjustments,
        auroc_version=criterion_config['auroc_gdro'],
        class_map=trainset.get_class_map('subclass'),
        use_cuda=use_cuda,
    )
    return criterion
