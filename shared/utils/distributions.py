from typing import Sequence

import numpy as np
import torch
from torch import Tensor
import torch.distributions as td

__all__ = ["DLogistic", "MixtureDistribution", "logistic_distribution", "uniform_bernoulli"]


def logistic_distribution(loc: Tensor, scale: Tensor):
    base_distribution = td.Uniform(loc.new_zeros(1), scale.new_zeros(1))
    transforms = [td.SigmoidTransform().inv, td.AffineTransform(loc=loc, scale=scale)]
    return td.TransformedDistribution(base_distribution, transforms)


class DLogistic(td.Distribution):
    def __init__(self, loc, scale):
        super().__init__()
        self.loc = loc
        self.scale = scale

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return self.scale ** 2

    def log_prob(self, value):
        upper = ((value + 0.5 - self.loc) / self.scale).sigmoid()
        lower = ((value - 0.5 - self.loc) / self.scale).sigmoid()
        return upper - lower


class MixtureDistribution(td.Distribution):
    def __init__(self, probs: Sequence[float], components: Sequence[td.Distribution]):
        assert len(probs) == len(components)
        assert all(prob >= 0 for prob in probs)
        assert sum(probs) == 1.0

        super().__init__()
        self.probs = probs
        self.components = components

    def log_prob(self, value):
        log_prob = 0.0
        for prob, dist in zip(self.probs, self.components):
            log_prob += prob * dist.log_prob(value)
        return log_prob


def uniform_bernoulli(shape, prob_1=0.5):
    nelement = int(np.product(shape))
    bern = td.Bernoulli(probs=prob_1)
    indexes = bern.sample((nelement,)).to(torch.bool)
    samples = torch.ones(nelement)

    ones = samples[indexes]
    ones.uniform_(0.5, 1.0)
    zeros = samples[~indexes]
    zeros.uniform_(0, 0.5)

    samples[indexes] = ones
    samples[~indexes] = zeros

    samples = samples.view(shape)

    return samples
