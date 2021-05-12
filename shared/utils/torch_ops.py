import torch
from torch import Tensor, jit
import torch.distributions as td
from torch.nn import functional as F

__all__ = [
    "RoundSTE",
    "logit",
    "normalized_softmax",
    "sum_except_batch",
    "to_discrete",
    "sample_concrete",
    "dot_product",
]


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: Tensor):
        return inputs.round()

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """Straight-through estimator"""
        return grad_output


def to_discrete(inputs: Tensor, dim: int = 1) -> Tensor:
    if inputs.dim() <= 1 or inputs.size(1) <= 1:
        return inputs.round()
    else:
        argmax = inputs.argmax(dim=1)
        return F.one_hot(argmax, num_classes=inputs.size(1))


def sample_concrete(logits: Tensor, temperature: float) -> Tensor:
    """Sample from the concrete/gumbel softmax distribution for
    differentiable discretization.
    Args:
        logits (Tensor): Logits to be transformed.
        temperature (float): Temperature of the distribution. The lower
        the temperature the closer the distribution comes to approximating
        a discrete distribution.
    Returns:
        Tensor: Samples from a concrete distribution with the
        given temperature.
    """
    if logits.dim() <= 1 or logits.size(1) <= 1:
        Concrete = td.RelaxedBernoulli
    else:
        Concrete = td.RelaxedOneHotCategorical
    concrete = Concrete(logits=logits, temperature=temperature)
    return concrete.rsample()


def logit(p: Tensor, eps: float = 1e-8) -> Tensor:
    p = p.clamp(min=eps, max=1.0 - eps)
    return torch.log(p / (1.0 - p))


def sum_except_batch(x: Tensor, keepdim: bool = False) -> Tensor:
    return x.flatten(start_dim=1).sum(-1, keepdim=keepdim)


def dot_product(x: Tensor, y: Tensor, keepdim: bool = False) -> Tensor:
    return torch.sum(x * y, dim=-1, keepdim=keepdim)


@jit.script
def normalized_softmax(logits: Tensor) -> Tensor:
    max_logits, _ = logits.max(dim=1, keepdim=True)
    unnormalized = torch.exp(logits - max_logits)
    return unnormalized / unnormalized.norm(p=2, dim=-1, keepdim=True)
