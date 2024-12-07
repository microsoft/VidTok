from typing import Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .util import default, instantiate_from_config, print0

from .discriminator import NLayerDiscriminator, NLayerDiscriminator3D, weights_init
from .lpips import LPIPS


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (torch.mean(F.softplus(-logits_real)) + torch.mean(F.softplus(logits_fake)))
    return d_loss


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight


def _sigmoid_cross_entropy_with_logits(labels, logits):
    """
    non-saturating loss
    """
    zeros = torch.zeros_like(logits, dtype=logits.dtype)
    condition = logits >= zeros
    relu_logits = torch.where(condition, logits, zeros)
    neg_abs_logits = torch.where(condition, -logits, logits)
    return relu_logits - logits * labels + torch.log1p(torch.exp(neg_abs_logits))


def non_saturate_gen_loss(logits_fake):
    """
    logits_fake: [B 1 H W]
    """
    B = logits_fake.shape[0]
    logits_fake = logits_fake.reshape(B, -1)
    logits_fake = torch.mean(logits_fake, dim=-1)
    gen_loss = torch.mean(_sigmoid_cross_entropy_with_logits(labels=torch.ones_like(logits_fake), logits=logits_fake))
    return gen_loss


def lecam_reg(real_pred, fake_pred, lecam_ema):
    reg = torch.mean(F.relu(real_pred - lecam_ema.logits_fake_ema).pow(2)) + torch.mean(
        F.relu(lecam_ema.logits_real_ema - fake_pred).pow(2)
    )
    return reg


class LeCAM_EMA(object):
    # https://github.com/TencentARC/SEED-Voken/blob/main/src/Open_MAGVIT2/modules/losses/vqperceptual.py
    def __init__(self, init=0.0, decay=0.999):
        self.logits_real_ema = init
        self.logits_fake_ema = init
        self.decay = decay

    def update(self, logits_real, logits_fake):
        self.logits_real_ema = self.logits_real_ema * self.decay + torch.mean(logits_real).item() * (1 - self.decay)
        self.logits_fake_ema = self.logits_fake_ema * self.decay + torch.mean(logits_fake).item() * (1 - self.decay)


class GeneralLPIPSWithDiscriminator(nn.Module):
    def __init__(
        self,
        disc_start: int,
        logvar_init: float = 0.0,
        pixelloss_weight=1.0,
        disc_num_layers: int = 3,
        disc_in_channels: int = 3,
        disc_factor: float = 1.0,
        disc_weight: float = 1.0,
        disc_type: str = "3d",
        perceptual_weight: float = 1.0,
        lecam_loss_weight: float = 0.0,
        disc_loss: str = "hinge",
        scale_input_to_tgt_size: bool = False,
        dims: int = 2,
        learn_logvar: bool = False,
        regularization_weights: Union[None, dict] = None,
        gen_loss_cross_entropy: bool = False,
    ):
        super().__init__()
        self.dims = dims
        if self.dims > 2:
            print0(
                f"[bold cyan]\[vidtok.modules.losses][GeneralLPIPSWithDiscriminator][/bold cyan] running with dims={dims}. This means that for perceptual loss calculation, "
                f"the LPIPS loss will be applied to each frame independently. "
            )
        self.scale_input_to_tgt_size = scale_input_to_tgt_size
        assert disc_loss in ["hinge", "vanilla"]
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        self.learn_logvar = learn_logvar
        self.disc_type = disc_type
        assert self.disc_type in ["2d", "3d"]

        if self.disc_type == "2d":
            self.discriminator = NLayerDiscriminator(
                input_nc=disc_in_channels, n_layers=disc_num_layers, use_actnorm=False
            ).apply(weights_init)
        else:
            self.discriminator = NLayerDiscriminator3D(
                input_nc=disc_in_channels, n_layers=disc_num_layers, use_actnorm=False
            ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.regularization_weights = default(regularization_weights, {})
        self.gen_loss_cross_entropy = gen_loss_cross_entropy
        self.lecam_loss_weight = lecam_loss_weight
        if self.lecam_loss_weight > 0:
            self.lecam_ema = LeCAM_EMA()

    def get_trainable_parameters(self) -> Any:
        return self.discriminator.parameters()

    def get_trainable_autoencoder_parameters(self) -> Any:
        if self.learn_logvar:
            yield self.logvar
        yield from ()

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(
        self,
        regularization_log,
        inputs,
        reconstructions,
        optimizer_idx,
        global_step,
        last_layer=None,
        split="train",
        weights=None,
    ):
        if self.scale_input_to_tgt_size:
            inputs = torch.nn.functional.interpolate(inputs, reconstructions.shape[2:], mode="bicubic", antialias=True)

        if optimizer_idx == 0:
            bs = inputs.shape[0]
            t = inputs.shape[2]
            if self.dims > 2:
                inputs, reconstructions = map(
                    lambda x: rearrange(x, "b c t h w -> (b t) c h w"),
                    (inputs, reconstructions),
                )

            rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
            if self.perceptual_weight > 0:
                p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
                rec_loss = rec_loss + self.perceptual_weight * p_loss
            else:
                p_loss = torch.Tensor([0.0])

            nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
            weighted_nll_loss = nll_loss
            if weights is not None:
                weighted_nll_loss = weights * nll_loss
            weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
            nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

            # now the GAN part
            if self.disc_type == "3d":
                reconstructions = rearrange(reconstructions, "(b t) c h w -> b c t h w", t=t).contiguous()

            # generator update
            logits_fake = self.discriminator(reconstructions)

            if not self.gen_loss_cross_entropy:
                g_loss = -torch.mean(logits_fake)
            else:
                g_loss = non_saturate_gen_loss(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + d_weight * disc_factor * g_loss
            log = dict()
            for k in regularization_log:
                if k in self.regularization_weights:
                    loss = loss + self.regularization_weights[k] * regularization_log[k]
                    log[f"{split}/{k}"] = regularization_log[k].detach().mean()

            log.update(
                {
                    "{}/total_loss".format(split): loss.clone().detach().mean(),
                    "{}/logvar".format(split): self.logvar.detach(),
                    "{}/nll_loss".format(split): nll_loss.detach().mean(),
                    "{}/rec_loss".format(split): rec_loss.detach().mean(),
                    "{}/p_loss".format(split): p_loss.detach().mean(),
                    "{}/d_weight".format(split): d_weight.detach(),
                    "{}/disc_factor".format(split): torch.tensor(disc_factor),
                    "{}/g_loss".format(split): g_loss.detach().mean(),
                }
            )
            return loss, log

        if optimizer_idx == 1:
            if self.disc_type == "2d" and self.dims > 2:
                inputs, reconstructions = map(
                    lambda x: rearrange(x, "b c t h w -> (b t) c h w"),
                    (inputs, reconstructions),
                )

            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)

            non_saturate_d_loss = self.disc_loss(logits_real, logits_fake)

            if self.lecam_loss_weight > 0:
                self.lecam_ema.update(logits_real, logits_fake)
                lecam_loss = lecam_reg(logits_real, logits_fake, self.lecam_ema)
                d_loss = disc_factor * (lecam_loss * self.lecam_loss_weight + non_saturate_d_loss)
            else:
                d_loss = disc_factor * non_saturate_d_loss

            log = {
                "{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                "{}/logits_real".format(split): logits_real.detach().mean(),
                "{}/logits_fake".format(split): logits_fake.detach().mean(),
                "{}/disc_factor".format(split): torch.tensor(disc_factor),
                "{}/non_saturated_d_loss".format(split): non_saturate_d_loss.detach(),
            }

            if self.lecam_loss_weight > 0:
                log.update({"{}/lecam_loss".format(split): lecam_loss.detach()})

            return d_loss, log
