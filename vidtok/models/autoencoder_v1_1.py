import re
from abc import abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Tuple, Union
from omegaconf import ListConfig
from packaging import version

import torch
import lightning.pytorch as pl

from safetensors.torch import load_file as load_safetensors
from vidtok.modules.ema import LitEma
from vidtok.modules.util import (default, get_obj_from_str,
                                 instantiate_from_config, print0)
from vidtok.modules.regularizers import pack_one, unpack_one, rearrange


class AbstractAutoencoder(pl.LightningModule):
    """
    This is the base class for all autoencoders
    """

    def __init__(
        self,
        ema_decay: Union[None, float] = None,
        monitor: Union[None, str] = None,
        mode: Union[None, str] = None,
        input_key: str = "jpg",
    ):
        super().__init__()

        self.input_key = input_key
        self.use_ema = ema_decay is not None
        self.ema_decay = ema_decay
        if monitor is not None:
            self.monitor = monitor
        if mode is not None:
            self.mode = mode

        if version.parse(torch.__version__) >= version.parse("2.0.0"):
            self.automatic_optimization = False

    @abstractmethod
    def init_from_ckpt(self, path: str, ignore_keys: Union[Tuple, list, ListConfig] = tuple(), verbose: bool = True) -> None:
        raise NotImplementedError()

    @abstractmethod
    def get_input(self, batch) -> Any:
        raise NotImplementedError()

    def on_train_batch_end(self, *args, **kwargs):
        # for EMA computation
        if self.use_ema:
            self.model_ema(self)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print0(
                    f"[bold magenta]\[vidtok.models.autoencoder][AbstractAutoencoder][/bold magenta] {context}: Switched to EMA weights"
                )
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print0(
                        f"[bold magenta]\[vidtok.models.autoencoder][AbstractAutoencoder][/bold magenta] {context}: Restored training weights"
                    )

    @abstractmethod
    def encode(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError(
            "[bold magenta]\[vidtok.models.autoencoder][AbstractAutoencoder][/bold magenta] encode()-method of abstract base class called"
        )

    @abstractmethod
    def decode(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError(
            "[bold magenta]\[vidtok.models.autoencoder][AbstractAutoencoder][/bold magenta] decode()-method of abstract base class called"
        )

    def instantiate_optimizer_from_config(self, params, lr, cfg):
        print0(
            f"[bold magenta]\[vidtok.models.autoencoder][AbstractAutoencoder][/bold magenta] loading >>> {cfg['target']} <<< optimizer from config"
        )
        return get_obj_from_str(cfg["target"])(params, lr=lr, **cfg.get("params", dict()))

    @abstractmethod
    def configure_optimizers(self) -> Any:
        raise NotImplementedError()


class AutoencodingEngine(AbstractAutoencoder):
    """
    Base class for all video tokenizers that we train
    """

    def __init__(
        self,
        *args,
        encoder_config: Dict,
        decoder_config: Dict,
        loss_config: Dict,
        regularizer_config: Dict,
        optimizer_config: Union[Dict, None] = None,
        lr_g_factor: float = 1.0,
        compile_model: bool = False,
        **kwargs,
    ):
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", ())
        verbose = kwargs.pop("verbose", True)
        self.use_tiling = kwargs.pop("use_tiling", False)
        self.t_chunk_enc = kwargs.pop("t_chunk_enc", 16)
        super().__init__(*args, **kwargs)

        compile = (
            torch.compile
            if (version.parse(torch.__version__) >= version.parse("2.0.0")) and compile_model
            else lambda x: x
        )

        self.encoder = compile(instantiate_from_config(encoder_config))
        self.decoder = compile(instantiate_from_config(decoder_config))
        self.loss = instantiate_from_config(loss_config)
        self.regularization = instantiate_from_config(regularizer_config)
        self.optimizer_config = default(optimizer_config, {"target": "torch.optim.Adam"})
        self.lr_g_factor = lr_g_factor

        self.t_chunk_dec = self.t_chunk_enc // self.encoder.time_downsample_factor
        self.use_overlap = False
        self.is_causal = self.encoder.is_causal

        if self.use_ema:
            self.model_ema = LitEma(self, decay=self.ema_decay)
            print0(
                f"[bold magenta]\[vidtok.models.autoencoder][AutoencodingEngine][/bold magenta] Keeping EMAs of {len(list(self.model_ema.buffers()))}."
            )

        print0(
            f"[bold magenta]\[vidtok.models.autoencoder][AutoencodingEngine][/bold magenta] Use ckpt_path: {ckpt_path}"
        )
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, verbose=verbose)

    def init_from_ckpt(self, path: str, ignore_keys: Union[Tuple, list, ListConfig] = tuple(), verbose: bool = True) -> None:
        if path.endswith("ckpt"):
            ckpt = torch.load(path, map_location="cpu")
            weights = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        elif path.endswith("safetensors"):
            weights = load_safetensors(path)
        else:
            raise NotImplementedError(f"Unknown checkpoint: {path}")

        keys = list(weights.keys())
        for k in keys:
            for ik in ignore_keys:
                if re.match(ik, k):
                    print0(
                        f"[bold magenta]\[vidtok.models.autoencoder][AutoencodingEngine][/bold magenta] Deleting key {k} from state_dict."
                    )
                    del weights[k]

        missing, unexpected = self.load_state_dict(weights, strict=False)
        print0(
            f"[bold magenta]\[vidtok.models.autoencoder][AutoencodingEngine][/bold magenta] Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if verbose:
            if len(missing) > 0:
                print0(
                    f"[bold magenta]\[vidtok.models.autoencoder][AutoencodingEngine][/bold magenta] Missing Keys: {missing}"
                )
            if len(unexpected) > 0:
                print0(
                    f"[bold magenta]\[vidtok.models.autoencoder][AutoencodingEngine][/bold magenta] Unexpected Keys: {unexpected}"
                )

    def get_input(self, batch: Dict) -> torch.Tensor:
        return batch[self.input_key]

    def get_autoencoder_params(self) -> list:
        params = (
            list(filter(lambda p: p.requires_grad, self.encoder.parameters()))
            + list(filter(lambda p: p.requires_grad, self.decoder.parameters()))
            + list(self.regularization.get_trainable_parameters())
            + list(self.loss.get_trainable_autoencoder_parameters())
        )
        return params

    def get_discriminator_params(self) -> list:
        params = list(self.loss.get_trainable_parameters())
        return params

    def get_last_layer(self):
        return self.decoder.get_last_layer()

    def _empty_causal_cached(self, parent):
        for name, module in parent.named_modules():
            if hasattr(module, 'causal_cache'):
                module.causal_cache = None

    def _set_first_chunk(self, is_first_chunk=True):
        for module in self.modules():
            if hasattr(module, 'is_first_chunk'):
                module.is_first_chunk = is_first_chunk

    def _set_cache_offset(self, modules, cache_offset=0):
        for module in modules:
            for submodule in module.modules():
                if hasattr(submodule, 'cache_offset'):
                    submodule.cache_offset = cache_offset

    def build_chunk_start_end(self, t, decoder_mode=False):
        start_end = [[0, 1]]
        start = 1
        end = start
        while True:
            if start >= t:
                break
            end = min(t, end + (self.t_chunk_dec if decoder_mode else self.t_chunk_enc))
            start_end.append([start, end])
            start = end
        return start_end

    def encode(self, x: Any, return_reg_log: bool = False) -> Any:
        self._empty_causal_cached(self.encoder)
        self._set_first_chunk(True)

        if self.use_tiling:
            z, reg_log = self.tile_encode(x)
        else:
            z = self.encoder(x)
            z, reg_log = self.regularization(z, n_steps=self.global_step // 2)

        if return_reg_log:
            return z, reg_log
        return z

    def tile_encode(self, x: Any) -> Any:
        num_frames = x.shape[2]
        start_end = self.build_chunk_start_end(num_frames)
        result_z, result_log = [], []

        for idx, (start, end) in enumerate(start_end):
            self._set_first_chunk(idx == 0)
            chunk = x[:, :, start:end, :, :]
            chunk_z = self.encoder(chunk)
            chunk_z, chunk_reg_log = self.regularization(chunk_z, n_steps=self.global_step//2)
            result_z.append(chunk_z)
            result_log.append(chunk_reg_log)
        try:
            kl_losses = [d['kl_loss'] for d in result_log]
            mean_kl_loss = torch.mean(torch.stack(kl_losses))
            return torch.cat(result_z, dim=2), {'kl_loss': mean_kl_loss}
        except:
            entropy_aux_losses = [d['aux_loss'] for d in result_log]
            mean_entropy_aux_loss = torch.mean(torch.stack(entropy_aux_losses))
            indices = [d['indices'] for d in result_log]
            return torch.cat(result_z, dim=2), {'aux_loss': mean_entropy_aux_loss, 'indices': torch.cat(indices, dim=1)}

    def indices_to_latent(self, token_indices: torch.Tensor) -> torch.Tensor:
        token_indices = rearrange(token_indices, "... -> ... 1")
        token_indices, ps = pack_one(token_indices, "b * d")
        codes = self.regularization.indices_to_codes(token_indices)
        codes = rearrange(codes, "b d n c -> b n (c d)")
        z = self.regularization.project_out(codes)
        z = unpack_one(z, ps, "b * d")
        z = rearrange(z, "b ... d -> b d ...")
        return z

    def tile_indices_to_latent(self, token_indices: torch.Tensor) -> torch.Tensor:
        num_frames = token_indices.shape[1]
        start_end = self.build_chunk_start_end(num_frames, decoder_mode=True)
        result_z = []
        for (start, end) in start_end:
            chunk = token_indices[:, start:end, :, :]
            chunk_z = self.indices_to_latent(chunk)
            result_z.append(chunk_z.clone())
        return torch.cat(result_z, dim=2)

    def decode(self, z: Any, decode_from_indices: bool = False) -> torch.Tensor:
        if decode_from_indices:
            if self.use_tiling:
                z = self.tile_indices_to_latent(z)
            else:
                z = self.indices_to_latent(z)

        self._empty_causal_cached(self.decoder)
        self._set_first_chunk(True)

        if self.use_tiling:
            x = self.tile_decode(z)
        else:
            x = self.decoder(z)
        return x

    def tile_decode(self, z: Any) -> torch.Tensor:
        num_frames = z.shape[2]
        start_end = self.build_chunk_start_end(num_frames, decoder_mode=True)
        result = []

        if self.use_overlap:
            assert self.encoder.time_downsample_factor in [2, 4, 8], "Only support 2x, 4x or 8x temporal downsampling now."
            if self.encoder.time_downsample_factor == 4:
                self._set_cache_offset([self.decoder], 1)
                self._set_cache_offset([self.decoder.up_temporal[2].upsample, self.decoder.up_temporal[1]], 2)
                self._set_cache_offset([self.decoder.up_temporal[1].upsample, self.decoder.up_temporal[0], self.decoder.conv_out], 4)
            elif self.encoder.time_downsample_factor == 2:
                self._set_cache_offset([self.decoder], 1)
                self._set_cache_offset([self.decoder.up_temporal[2].upsample, self.decoder.up_temporal[1], self.decoder.up_temporal[0], self.decoder.conv_out], 2)
            else:
                self._set_cache_offset([self.decoder], 1)
                self._set_cache_offset([self.decoder.up_temporal[3].upsample, self.decoder.up_temporal[2]], 2)
                self._set_cache_offset([self.decoder.up_temporal[2].upsample, self.decoder.up_temporal[1]], 4)
                self._set_cache_offset([self.decoder.up_temporal[1].upsample, self.decoder.up_temporal[0], self.decoder.conv_out], 8)

        for idx, (start, end) in enumerate(start_end):
            self._set_first_chunk(idx == 0)
            chunk_z = z[:, :, start:end+1, :, :] if self.use_overlap and end + 1 <= num_frames else z[:, :, start:end, :, :]
            chunk = self.decoder(chunk_z)

            if self.use_overlap and end + 1 <= num_frames:
                chunk = chunk[:, :, :-self.encoder.time_downsample_factor]
            result.append(chunk.clone())

        return torch.cat(result, dim=2)

    def forward(self, x: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.encoder.fix_encoder:
            with torch.no_grad():
                z, reg_log = self.encode(x, return_reg_log=True)
        else:
            z, reg_log = self.encode(x, return_reg_log=True)
        dec = self.decode(z)
        if dec.shape[2] != x.shape[2]:
            dec = dec[:, :, -x.shape[2]:, ...]
        return z, dec, reg_log

    def training_step(self, batch, batch_idx) -> Any:
        x = self.get_input(batch)

        if x.ndim == 4:
            x = x.unsqueeze(2)

        z, xrec, regularization_log = self(x)

        if x.ndim == 5 and xrec.ndim == 4:
            xrec = xrec.unsqueeze(2)

        opt_g, opt_d = self.optimizers()

        # autoencode loss
        self.toggle_optimizer(opt_g)
        aeloss, log_dict_ae = self.loss(
            regularization_log,
            x,
            xrec,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )
        opt_g.zero_grad()
        self.manual_backward(aeloss)

        # gradient clip
        torch.nn.utils.clip_grad_norm_(self.get_autoencoder_params(), 20.0)
        opt_g.step()
        self.untoggle_optimizer(opt_g)

        # discriminator loss
        self.toggle_optimizer(opt_d)
        discloss, log_dict_disc = self.loss(
            regularization_log,
            x,
            xrec,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )
        opt_d.zero_grad()
        self.manual_backward(discloss)
        torch.nn.utils.clip_grad_norm_(self.get_discriminator_params(), 20.0)
        opt_d.step()
        self.untoggle_optimizer(opt_d)

        # logging
        log_dict = {
            "train/aeloss": aeloss,
            "train/discloss": discloss,
        }
        log_dict.update(log_dict_ae)
        log_dict.update(log_dict_disc)

        self.log_dict(log_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        lr = opt_g.param_groups[0]["lr"]
        self.log(
            "lr_abs",
            lr,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

    def validation_step(self, batch, batch_idx) -> Dict:
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, postfix="_ema")
            log_dict.update(log_dict_ema)
        return log_dict

    def _validation_step(self, batch, batch_idx, postfix="") -> Dict:
        x = self.get_input(batch)

        if x.ndim == 4:
            x = x.unsqueeze(2)

        z, xrec, regularization_log = self(x)

        if x.ndim == 5 and xrec.ndim == 4:
            xrec = xrec.unsqueeze(2)

        aeloss, log_dict_ae = self.loss(
            regularization_log,
            x,
            xrec,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val" + postfix,
        )

        discloss, log_dict_disc = self.loss(
            regularization_log,
            x,
            xrec,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val" + postfix,
        )

        self.log(f"val{postfix}/rec_loss", log_dict_ae[f"val{postfix}/rec_loss"])
        log_dict_ae.update(log_dict_disc)
        self.log_dict(log_dict_ae)
        return log_dict_ae

    def configure_optimizers(self) -> Any:
        ae_params = self.get_autoencoder_params()
        disc_params = self.get_discriminator_params()

        opt_ae = self.instantiate_optimizer_from_config(
            ae_params,
            default(self.lr_g_factor, 1.0) * self.learning_rate,
            self.optimizer_config,
        )
        opt_disc = self.instantiate_optimizer_from_config(disc_params, self.learning_rate, self.optimizer_config)

        return [opt_ae, opt_disc], []

    @torch.no_grad()
    def log_images(self, batch: Dict) -> Dict:
        log = dict()
        x = self.get_input(batch)
        _, xrec, _ = self(x)
        log["inputs"] = x
        log["recs"] = xrec
        with self.ema_scope():
            _, xrec_ema, _ = self(x)
            log["recs_ema"] = xrec_ema
        return log
