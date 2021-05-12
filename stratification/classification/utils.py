from sklearn.metrics import roc_auc_score
import torch


class AverageMeter:
    """Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = [0] * 4

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.data.cpu().item()
        if isinstance(n, torch.Tensor):
            n = n.data.cpu().item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0

    def __str__(self):
        return f'sum: {self.sum}, count: {self.count}, avg: {self.avg}'


def compute_accuracy(output, target, topk=(1,), return_preds=False):
    """Computes the precision@k for the specified values of k"""
    topk_orig = topk
    topk = [k for k in topk if k <= output.size(1)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    label_preds = pred[:, 0]
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    res.extend(100 for k in topk_orig if k > output.size(1))
    if return_preds:
        return res, label_preds
    return res


def compute_roc_auc(targets, probs):
    """'Safe' AUROC computation"""
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()
    if isinstance(probs, torch.Tensor):
        probs = probs.numpy()
    try:
        auroc = roc_auc_score(targets, probs)
    except ValueError:
        auroc = -1
    return auroc
