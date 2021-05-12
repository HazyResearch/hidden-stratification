from __future__ import annotations
from typing import Any, overload

import torch
from torch import Tensor
from torch.cuda.amp.grad_scaler import GradScaler
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing_extensions import Literal

from shared.configs import VaeStd, ZsTransform
from shared.utils import RoundSTE, sample_concrete, to_discrete

from .base import (
    EncodingSize,
    ModelBase,
    Reconstructions,
    SplitDistributions,
    SplitEncoding,
)

__all__ = ["AutoEncoder", "Vae"]


class AutoEncoder(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        encoding_size: EncodingSize | None,
        zs_transform: ZsTransform = ZsTransform.none,
        feature_group_slices: dict[str, list[slice]] | None = None,
        optimizer_kwargs: dict[str, Any] | None = None,
    ):
        super(AutoEncoder, self).__init__()

        self.encoder: ModelBase = ModelBase(encoder, optimizer_kwargs=optimizer_kwargs)
        self.decoder: ModelBase = ModelBase(decoder, optimizer_kwargs=optimizer_kwargs)
        self.encoding_size = encoding_size
        self.feature_group_slices = feature_group_slices
        self.zs_transform = zs_transform

    def encode(self, inputs: Tensor, *, stochastic: bool = False) -> SplitEncoding:
        del stochastic
        enc = self._split_encoding(self.encoder(inputs))
        if self.zs_transform is ZsTransform.round_ste:
            rounded_zs = RoundSTE.apply(torch.sigmoid(enc.zs))
        else:
            rounded_zs = enc.zs
        return SplitEncoding(zs=rounded_zs, zy=enc.zy)

    def decode(
        self, enc: SplitEncoding, mode: Literal["soft", "hard", "relaxed"] = "soft"
    ) -> Tensor:
        decoding = self.decoder(self.unsplit_encoding(enc))
        if decoding.dim() == 4:
            # if decoding.size(1) <= 3:
            #     decoding = decoding.sigmoid()
            # else:
            if decoding.size(1) > 3:  # if we use CE losss, we have more than 3 channels
                # conversion for cross-entropy loss
                num_classes = 256
                decoding = decoding.view(decoding.size(0), num_classes, -1, *decoding.shape[-2:])
        else:
            if mode in ("hard", "relaxed") and self.feature_group_slices:
                discrete_outputs = []
                stop_index = 0
                #   Sample from discrete variables using the straight-through-estimator
                for group_slice in self.feature_group_slices["discrete"]:
                    if mode == "hard":
                        discrete_outputs.append(to_discrete(decoding[:, group_slice]).float())
                    else:
                        discrete_outputs.append(
                            sample_concrete(decoding[:, group_slice], temperature=1e-2)
                        )
                    stop_index = group_slice.stop
                discrete_outputs = torch.cat(discrete_outputs, axis=1)
                decoding = torch.cat([discrete_outputs, decoding[:, stop_index:]], axis=1)

        return decoding

    def all_recons(
        self, enc: SplitEncoding, mode: Literal["soft", "hard", "relaxed"]
    ) -> Reconstructions:
        rand_s, rand_y = self.mask(enc, random=True)
        zero_s, zero_y = self.mask(enc)
        just_s = SplitEncoding(zs=enc.zs, zy=torch.zeros_like(enc.zy))
        return Reconstructions(
            all=self.decode(enc, mode=mode),
            rand_s=self.decode(rand_s, mode=mode),
            rand_y=self.decode(rand_y, mode=mode),
            zero_s=self.decode(zero_s, mode=mode),
            zero_y=self.decode(zero_y, mode=mode),
            just_s=self.decode(just_s, mode=mode),
        )

    def forward(self, inputs):
        return self.encode(inputs)

    def zero_grad(self):
        self.encoder.zero_grad()
        self.decoder.zero_grad()

    def step(self, grad_scaler: GradScaler | None = None):
        self.encoder.step(grad_scaler=grad_scaler)
        self.decoder.step(grad_scaler=grad_scaler)

    def _split_encoding(self, z: Tensor) -> SplitEncoding:
        assert self.encoding_size is not None
        zs, zy = z.split((self.encoding_size.zs, self.encoding_size.zy), dim=1)
        return SplitEncoding(zs=zs, zy=zy)

    @staticmethod
    def unsplit_encoding(enc: SplitEncoding) -> Tensor:
        return torch.cat([enc.zs, enc.zy], dim=1)

    def mask(
        self, enc: SplitEncoding, random: bool = False, detach: bool = False
    ) -> tuple[SplitEncoding, SplitEncoding]:
        """Mask out zs and zy. This is a cheap function.

        Args:
            enc: encoding to mask
            random: whether to replace the masked out part with random noise
            detach: whether to detach from the computational graph before masking
        """
        zs = enc.zs
        zy = enc.zy
        if detach:
            zs = zs.detach()
            zy = zy.detach()
        if random:
            zs_m = SplitEncoding(zs=torch.randn_like(zs), zy=zy)
            zy_m = SplitEncoding(zs=zs, zy=torch.randn_like(zy))
        else:
            zs_m = SplitEncoding(zs=torch.zeros_like(zs), zy=zy)
            zy_m = SplitEncoding(zs=zs, zy=torch.zeros_like(zy))
        return zs_m, zy_m

    def fit(self, train_data: DataLoader, epochs: int, device, loss_fn, kl_weight: float):
        self.train()

        with tqdm(total=epochs * len(train_data)) as pbar:
            for _ in range(epochs):

                for x, _, _ in train_data:

                    x = x.to(device)

                    self.zero_grad()
                    _, loss, _ = self.routine(x, recon_loss_fn=loss_fn, kl_weight=kl_weight)
                    # loss /= x[0].nelement()

                    loss.backward()
                    self.step()

                    pbar.update()
                    pbar.set_postfix(AE_loss=loss.detach().cpu().numpy())

    def routine(
        self, x: Tensor, recon_loss_fn, kl_weight: float
    ) -> tuple[SplitEncoding, Tensor, dict[str, float]]:
        encoding = self.encode(x)

        recon_all = self.decode(encoding)
        recon_loss = recon_loss_fn(recon_all, x)
        recon_loss /= x.nelement()
        prior_loss = kl_weight * encoding.zy.norm(dim=1).mean()
        loss = recon_loss + prior_loss
        return encoding, loss, {"Loss reconstruction": recon_loss.item(), "Prior Loss": prior_loss}


class Vae(AutoEncoder):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        encoding_size: EncodingSize | None,
        vae_std_tform: VaeStd,
        feature_group_slices: dict[str, list[slice]] | None = None,
        optimizer_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            encoding_size=encoding_size,
            feature_group_slices=feature_group_slices,
            optimizer_kwargs=optimizer_kwargs,
        )
        self.encoder: ModelBase = ModelBase(encoder, optimizer_kwargs=optimizer_kwargs)
        self.decoder: ModelBase = ModelBase(decoder, optimizer_kwargs=optimizer_kwargs)

        self.prior = td.Normal(0, 1)
        self.posterior_fn = td.Normal
        self.vae_std_tform = vae_std_tform

    def compute_divergence(self, sample: Tensor, posterior: td.Distribution) -> Tensor:
        log_p = self.prior.log_prob(sample)
        log_q = posterior.log_prob(sample)

        return (log_q - log_p).sum()

    def _get_posterior(self, loc: Tensor, scale: Tensor) -> td.Distribution:
        if self.vae_std_tform == VaeStd.softplus:
            scale = F.softplus(scale)
        else:
            scale = torch.exp(0.5 * scale).clamp(min=0.005, max=3.0)
        return self.posterior_fn(loc, scale)

    @overload
    def encode(
        self, inputs: Tensor, *, stochastic: bool = ..., return_posterior: Literal[True]
    ) -> tuple[SplitEncoding, SplitDistributions]:
        ...

    @overload
    def encode(
        self, inputs: Tensor, *, stochastic: bool = ..., return_posterior: Literal[False] = ...
    ) -> SplitEncoding:
        ...

    def encode(
        self, inputs: Tensor, *, stochastic: bool = False, return_posterior: bool = False
    ) -> tuple[SplitEncoding, SplitDistributions] | SplitEncoding:
        loc_all, scale_all = self.encoder(inputs).chunk(2, dim=1)
        loc = self._split_encoding(loc_all)
        scale = self._split_encoding(scale_all)

        samples = []
        posteriors = []
        for loc_, scale_ in zip((loc.zs, loc.zy), (scale.zs, scale.zy)):
            if stochastic:
                posterior = self._get_posterior(loc_, scale_)
                samples.append(posterior.rsample())
                if return_posterior:
                    posteriors.append(posterior)
            else:
                samples.append(loc_)
                if return_posterior:
                    posteriors.append(self._get_posterior(loc_, scale_))

        sample = SplitEncoding(zs=samples[0], zy=samples[1])
        if return_posterior:
            return sample, SplitDistributions(zs=posteriors[0], zy=posteriors[1])
        else:
            return sample

    def routine(
        self, x: Tensor, recon_loss_fn, kl_weight: float
    ) -> tuple[SplitEncoding, Tensor, dict[str, float]]:
        encoding, posterior = self.encode(x, return_posterior=True, stochastic=True)
        kl_div = self.compute_divergence(encoding.zs, posterior.zs)
        kl_div += self.compute_divergence(encoding.zy, posterior.zy)
        kl_div /= x.nelement()
        kl_div *= kl_weight

        recon_all = self.decode(encoding)
        recon_loss = recon_loss_fn(recon_all, x)
        recon_loss /= x.nelement()
        elbo = recon_loss + kl_div
        logging_dict = {"Loss Reconstruction": recon_loss.item(), "KL divergence": kl_div}
        return encoding, elbo, logging_dict
