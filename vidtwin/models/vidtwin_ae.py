import os
import re
import math
from abc import abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Tuple, Union

import lightning.pytorch as pl
import torch
import einops
from omegaconf import ListConfig
from packaging import version
from safetensors.torch import load_file as load_safetensors

from torch.optim.lr_scheduler import _LRScheduler, LambdaLR, StepLR
from vidtok.modules.util import default, instantiate_from_config, print0, get_valid_paths
from vidtok.modules.util import compute_psnr, compute_ssim
from vidtok.models.autoencoder import AbstractAutoencoder
import numpy as np
from torch import nn
from einops import rearrange, repeat
import transformers


class VidAutoEncoderQformerBase(AbstractAutoencoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def init_from_ckpt(
        self, path: str, ignore_keys: Union[Tuple, list, ListConfig] = tuple()
    ) -> None:
        if path.endswith("ckpt"):
            # sd = torch.load(path, map_location="cpu")["state_dict"]
            ckpt = torch.load(path, map_location="cpu")
            if "state_dict" in ckpt:
                sd = ckpt["state_dict"]
            else:
                sd = ckpt
        elif path.endswith("safetensors"):
            sd = load_safetensors(path)
        else:
            raise NotImplementedError

        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if re.match(ik, k):
                    print0(f"[bold magenta]\[vidtok.models.vidtwin_ae][VidAutoencoderQformer][/bold magenta] Deleting key {k} from state_dict.")
                    del sd[k]
                    
        for k, tensor in sd.items():
            sd[k] = tensor.to(torch.float64)
            
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print0(
            f"[bold magenta]\[vidtok.models.vidtwin_ae][VidAutoencoderQformer][/bold magenta] Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print0(f"[bold magenta]\[vidtok.models.vidtwin_ae][VidAutoencoderQformer][/bold magenta] Missing Keys: {missing}")
        if len(unexpected) > 0:
            print0(f"[bold magenta]\[vidtok.models.vidtwin_ae][VidAutoencoderQformer][/bold magenta] Unexpected Keys: {unexpected}")
            
    def get_input(self, batch: Dict) -> torch.Tensor:
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in channels-first format (e.g., bchw instead if bhwc)
        return batch[self.input_key]   
         
    def get_autoencoder_params(self) -> list:
        params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.get_disentangle_params())
            + list(self.regularization.get_trainable_parameters())
            + list(self.loss.get_trainable_autoencoder_parameters())
        )
        return params

    def get_discriminator_params(self) -> list:
        params = list(self.loss.get_trainable_parameters())  # e.g., discriminator
        return params

    def get_last_layer(self):
        return self.decoder.get_last_layer()

    # See https://github.com/Lightning-AI/pytorch-lightning/issues/17801 and https://lightning.ai/docs/pytorch/stable/common/optimization.html for the reason of this change
    def training_step(self, batch, batch_idx) -> Any:   
        x = self.get_input(batch)
        z, xrec, regularization_log, *_ = self(x)
        opt_g, opt_d = self.optimizers()
        sch1, sch2 = self.lr_schedulers()
     

        # autoencode loss
        self.toggle_optimizer(opt_g)
        # adversarial loss is binary cross-entropy
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
        opt_g.step()
        sch1.step()
        self.untoggle_optimizer(opt_g)

        # discriminator loss
        self.toggle_optimizer(opt_d)
        # adversarial loss is binary cross-entropy
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
        opt_d.step()
           
        sch2.step()
        self.untoggle_optimizer(opt_d)

        # logging
        log_dict = {
            "train/aeloss": aeloss,
            "train/discloss": discloss,
        }
        log_dict.update(log_dict_ae)
        log_dict.update(log_dict_disc)
        self.log_dict(log_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)

    def validation_step(self, batch, batch_idx) -> Dict:
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, postfix="_ema")
            log_dict.update(log_dict_ema)
        return log_dict

    def _validation_step(self, batch, batch_idx, postfix="") -> Dict:
        x = self.get_input(batch)

        z, xrec, regularization_log, *_ = self(x)
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

        # evaluate the psnr and ssim
        x = x.clamp(-1, 1)
        xrec = xrec.clamp(-1, 1)
        x = (x + 1) / 2
        xrec = (xrec + 1) / 2
        psnr = compute_psnr(xrec, x)
        ssim = compute_ssim(xrec, x)

        self.log(f"val{postfix}/psnr", psnr, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log(f"val{postfix}/ssim", ssim, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return log_dict_ae

    def configure_optimizers(self):
        if self.trainable_ae_params is None:
            ae_params = self.get_autoencoder_params()
            print0(f"[bold magenta]\[vidtok.models.vidtwin_ae][VidAutoencoderQformer][/bold magenta] Number of trainable autoencoder parameters: {len(ae_params):,}")
        else:           
            ae_params, num_ae_params = self.get_param_groups(
                self.trainable_ae_params, self.ae_optimizer_args
            )
            print0(f"[bold magenta]\[vidtok.models.vidtwin_ae][VidAutoencoderQformer][/bold magenta] Number of trainable autoencoder parameters: {num_ae_params:,}")
        if self.trainable_disc_params is None:
            disc_params = self.get_discriminator_params()
            print0(f"[bold magenta]\[vidtok.models.vidtwin_ae][VidAutoencoderQformer][/bold magenta] Number of trainable discriminator parameters: {len(disc_params):,}")
        else:
            disc_params, num_disc_params = self.get_param_groups(
                self.trainable_disc_params, self.disc_optimizer_args
            )
            print0(
                f"[bold magenta]\[vidtok.models.vidtwin_ae][VidAutoencoderQformer][/bold magenta] Number of trainable discriminator parameters: {num_disc_params:,}"
            )
        opt_ae = self.instantiate_optimizer_from_config(
            ae_params,
            default(self.lr_g_factor, 1.0) * self.learning_rate,
            self.optimizer_config,
        )
        
        if len(disc_params) > 0:
            opt_disc = self.instantiate_optimizer_from_config(
                disc_params, self.learning_rate, self.optimizer_config
            )
        
        lr_freq1 = 1
        lr_freq2 = 1
        if not self.use_scheduler_g:
            total_steps = len(self.trainer.datamodule.train_dataloader()) * self.trainer.max_epochs
            scheduler1 = ConstantWarmupScheduler(opt_ae, warmup_steps=500, total_steps=total_steps)
        else:
            print0(f"[bold magenta]\[vidtok.models.vidtwin_ae][VidAutoencoderQformer][/bold magenta] Use generator lr scheduler: {self.lr_scheduler_config_g.target}")
            lr_freq1 = self.lr_scheduler_config_g.params.frequency if hasattr(self.lr_scheduler_config_g.params, 'frequency') else 1
            max_decay_steps = len(self.trainer.datamodule.train_dataloader()) * self.trainer.max_epochs
            print0(f"[bold magenta]\[vidtok.models.vidtwin_ae][VidAutoencoderQformer][/bold magenta] Use discriminator lr scheduler max_decay_steps: {max_decay_steps}")
            if 'inverse_sqrt' in self.lr_scheduler_config_g.target:
                scheduler1 = transformers.get_inverse_sqrt_schedule(optimizer=opt_ae, num_warmup_steps=self.lr_scheduler_config_g.params.num_warmup_steps)
            elif 'LambdaWarmUpCosineScheduler' in self.lr_scheduler_config_g.target:
                scheduler1 = LambdaWarmUpCosineScheduler(optimizer=opt_ae, total_steps=max_decay_steps, **self.lr_scheduler_config_g.params) 
            elif 'LinearWarmupScheduler' in self.lr_scheduler_config_g.target:
                scheduler1 = LinearWarmupScheduler(opt_ae, total_steps=max_decay_steps, **self.lr_scheduler_config_g.params)
            else:
                scheduler1 = instantiate_lrscheduler_from_config(opt_ae, self.lr_scheduler_config_g, total_steps=max_decay_steps)

        if not self.use_scheduler_d:
            total_steps = len(self.trainer.datamodule.train_dataloader()) * self.trainer.max_epochs
            scheduler2 = ConstantWarmupScheduler(opt_disc, warmup_steps=500, total_steps=total_steps)
        else:
            print0(f"[bold magenta]\[vidtok.models.vidtwin_ae][VidAutoencoderQformer][/bold magenta] Use discriminator lr scheduler: {self.lr_scheduler_config_d.target}")
            lr_freq2 = self.lr_scheduler_config_d.params.frequency if hasattr(self.lr_scheduler_config_d.params, 'frequency') else 1
            max_decay_steps = len(self.trainer.datamodule.train_dataloader()) * self.trainer.max_epochs
            print0(f"[bold magenta]\[vidtok.models.vidtwin_ae][VidAutoencoderQformer][/bold magenta] Use discriminator lr scheduler max_decay_steps: {max_decay_steps}")
            if 'inverse_sqrt' in self.lr_scheduler_config_d.target:
                scheduler2 = transformers.get_inverse_sqrt_schedule(optimizer=opt_disc, num_warmup_steps=self.lr_scheduler_config_d.params.num_warmup_steps)
            elif 'LambdaWarmUpCosineScheduler' in self.lr_scheduler_config_d.target:
                scheduler2 = LambdaWarmUpCosineScheduler(optimizer=opt_disc, total_steps=max_decay_steps, **self.lr_scheduler_config_d.params)
            elif 'LinearWarmupScheduler' in self.lr_scheduler_config_d.target:
                scheduler2 = LinearWarmupScheduler(opt_disc, total_steps=max_decay_steps,  **self.lr_scheduler_config_d.params)
            else:
                scheduler2 = instantiate_lrscheduler_from_config(opt_disc, self.lr_scheduler_config_d, total_steps=max_decay_steps)


        lr_scheduler_config1 = {
            "optimizer": opt_ae,
            "lr_scheduler": {
                "scheduler": scheduler1,
                "name": "lr_generator",
                "interval": "step",
                "frequency": lr_freq1,
            }
        }
        lr_scheduler_config2 = {
            "optimizer": opt_disc,
            "lr_scheduler": {
                "scheduler": scheduler2,
                "name": "lr_discriminator",
                "interval": "step",
                "frequency": lr_freq2,
            }
        }
        return (lr_scheduler_config1, lr_scheduler_config2)

    @torch.no_grad()
    def log_images(self, batch: Dict, **kwargs) -> Dict:   # called at ImageLoggerCallback.log_img()
        log = dict()
        x = self.get_input(batch)
        _, xrec, *_ = self(x)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log


class VidAutoEncoderQformer(VidAutoEncoderQformerBase):

    def __init__(
        self,
        *args,
        encoder_config: Dict,
        decoder_config: Dict,
        loss_config: Dict,
        regularizer_config: Dict,
        temporal_qformer_config: Dict,
        height_qformer_config: Dict,
        width_qformer_config: Dict,
        lr_scheduler_config_g=None,
        lr_scheduler_config_d=None,
        trainable_ae_params=None,
        ae_optimizer_args = None,
        trainable_disc_params = None,
        lr_scheduler_config: Dict = None,
        weight_decay: float = 1e-5,
        disc_optimizer_args = None,
        optimizer_config: Union[Dict, None] = None,
        lr_g_factor: float = 1.0,
        compile_model: bool = False,
        **kwargs,
    ):
        ckpt_path = kwargs.pop("ckpt_path", None)
        ckpt_path2 = kwargs.pop("ckpt_path2", None)
        ignore_keys = kwargs.pop("ignore_keys", ())
        super().__init__(*args, **kwargs)
        compile = (
            torch.compile
            if (version.parse(torch.__version__) >= version.parse("2.0.0"))
            and compile_model
            else lambda x: x
        )

        self.encoder = compile(instantiate_from_config(encoder_config))
        self.decoder = compile(instantiate_from_config(decoder_config))
        self.loss = instantiate_from_config(loss_config)
        self.regularization = instantiate_from_config(regularizer_config)
        
        # define the qformer
        self.temporal_qformer = instantiate_from_config(temporal_qformer_config)
        self.hight_qformer = instantiate_from_config(height_qformer_config)
        self.width_qformer = instantiate_from_config(width_qformer_config)
        
        
        
        self.use_scheduler = lr_scheduler_config is not None
        self.check = 0
        self.weight_decay = weight_decay
        if self.use_scheduler:
            self.lr_scheduler_config = lr_scheduler_config
        self.use_scheduler_g = lr_scheduler_config_g is not None 
        self.use_scheduler_d = lr_scheduler_config_d is not None
        if self.use_scheduler_g:
            self.lr_scheduler_config_g = lr_scheduler_config_g
        if self.use_scheduler_d:
            self.lr_scheduler_config_d = lr_scheduler_config_d
        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.Adam", "params": {"betas": (0, 0.99), "weight_decay": self.weight_decay}})
        self.trainable_ae_params = trainable_ae_params
        if self.trainable_ae_params is not None:
            self.ae_optimizer_args = default(
                ae_optimizer_args,
                [{} for _ in range(len(self.trainable_ae_params))],
            )
            assert len(self.ae_optimizer_args) == len(self.trainable_ae_params)
        else:
            self.ae_optimizer_args = [{}]  # makes type consitent
        self.trainable_disc_params = trainable_disc_params
        if self.trainable_disc_params is not None:
            self.disc_optimizer_args = default(
                disc_optimizer_args,
                [{} for _ in range(len(self.trainable_disc_params))],
            )
            assert len(self.disc_optimizer_args) == len(self.trainable_disc_params)
        else:
            self.disc_optimizer_args = [{}]  # makes type consitent
        
        self.lr_g_factor = lr_g_factor

        self.hidden_dim = encoder_config.params.hidden_size
        self.patch_nums = np.array(list(encoder_config.params.input_size)) // np.array(list(encoder_config.params.patch_size))
        # (bhw, f, c) -> (bhw, f',c')
        self.cont_emb = nn.Sequential(
            nn.Linear(temporal_qformer_config.params.query_hidden_size, self.hidden_dim),
            nn.ReLU(),
            nn.Conv1d(temporal_qformer_config.params.num_query_tokens, self.patch_nums[0], 1),
            nn.ReLU(),
        )
        
        self.height_emb = nn.Sequential(
            nn.Linear(height_qformer_config.params.query_hidden_size, self.hidden_dim),
            nn.ReLU(),
            nn.Conv1d(height_qformer_config.params.num_query_tokens, self.patch_nums[1], 1),
            nn.ReLU(),
        )
        
        self.width_emb = nn.Sequential(
            nn.Linear(width_qformer_config.params.query_hidden_size, self.hidden_dim),
            nn.ReLU(),
            nn.Conv1d(width_qformer_config.params.num_query_tokens, self.patch_nums[2], 1),
            nn.ReLU(),
        )
        
        ckpt_path = get_valid_paths(ckpt_path, ckpt_path2)
        print0(f"[bold magenta]\[vidtok.models.vidtwin_ae][VidAutoencoderQformer][/bold magenta] Use ckpt_path: {ckpt_path}")
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)


    
    def get_disentangle_params(self) -> list:
        params = (
            list(self.temporal_qformer.parameters())
            + list(self.hight_qformer.parameters())
            + list(self.width_qformer.parameters())
            + list(self.cont_emb.parameters())
            + list(self.height_emb.parameters())
            + list(self.width_emb.parameters())
        )
            
        return params
    

    def decode(self, z,  z_content, z_motion_x, z_motion_y) -> torch.Tensor:
        '''
        input: z: shape (b, c', f, h', w')
                z_content: shape (b, f_q, h', w', c_q)
                z_motion_x: shape (b, f, h_q, w', c_q)
                z_motion_y: shape (b, f, h', w_q, c_q)
        '''
        z_content = rearrange(z_content, 'B F H W C -> (B H W) F C')
        vt = rearrange(self.cont_emb(z_content), '(B H W) F C -> B C F H W', H=z.size(3), W=z.size(4))
        z_motion_x = rearrange(z_motion_x, 'B F H W C -> (B F W) H C')
        vx = rearrange(self.height_emb(z_motion_x), '(B F W) H C -> B C F H W', F=z.size(2), W=z.size(4))
        z_motion_y = rearrange(z_motion_y, 'B F H W C -> (B F H) W C')
        vy = rearrange(self.width_emb(z_motion_y), '(B F H) W C -> B C F H W', F=z.size(2), H=z.size(3))
        c_plus_m = vt + vx + vy # shape (b, c', f, h', w')
        x = self.decoder(c_plus_m)
        return x

    def encode(self, x: Any, return_reg_log: bool = False) -> Any:
        z = self.encoder(x) # shape (b, c', f, h', w')
        z_content = self.temporal_qformer(rearrange(z, 'B C F H W -> (B H W) F C')) 
        z_content = rearrange(z_content, '(B H W) F C -> B F H W C', H=z.size(3), W=z.size(4)) # compressed in the temporal dimension
        z_motion_x = self.hight_qformer(rearrange(z, 'B C F H W -> (B F W) H C'))
        z_motion_x = rearrange(z_motion_x, '(B F W) H C -> B F H W C', F=z.size(2), W=z.size(4)) # compressed in the height dimension
        z_motion_y = self.width_qformer(rearrange(z, 'B C F H W -> (B F H) W C'))
        z_motion_y = rearrange(z_motion_y, '(B F H) W C -> B F H W C', F=z.size(2), H=z.size(3)) # compressed in the width dimension
        if return_reg_log:
            return z, z_content, z_motion_x, z_motion_y, None
        return z, z_content, z_motion_x, z_motion_y

    
    def forward(self, x: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (bs, 3, 17, h, w)
        z, z_content, z_motion_x, z_motion_y, reg_log = self.encode(x, return_reg_log=True)
        # z: shape (b, c', f, h', w')
        dec = self.decode(z, z_content, z_motion_x, z_motion_y)
        # dec: (bs, 3, 17, h, w)
        return z, dec, reg_log, z_content, z_motion_x, z_motion_y



class VidAutoEncoderQformerCompact(VidAutoEncoderQformerBase):
    
    def __init__(
        self,
        *args,
        encoder_config: Dict,
        decoder_config: Dict,
        loss_config: Dict,
        regularizer_config: Dict,
        temporal_qformer_config: Dict,
        space_qformer_config: Dict,
        lr_scheduler_config_g=None,
        lr_scheduler_config_d=None,
        trainable_ae_params=None,
        ae_optimizer_args = None,
        trainable_disc_params = None,
        lr_scheduler_config: Dict = None,
        weight_decay: float = 1e-5,
        disc_optimizer_args = None,
        optimizer_config: Union[Dict, None] = None,
        lr_g_factor: float = 1.0,
        compile_model: bool = False,
        retain_num_frames: bool = True,
        temporal_down_dim: int = 32,
        partial_content_motion: str = 'all',
        shuffle_content: bool = False,
        repeat_for_decoder: bool = False,
        **kwargs,
    ):
        ckpt_path = kwargs.pop("ckpt_path", None)
        ckpt_path2 = kwargs.pop("ckpt_path2", None)
        ignore_keys = kwargs.pop("ignore_keys", ())
        super().__init__(*args, **kwargs)
        compile = (
            torch.compile
            if (version.parse(torch.__version__) >= version.parse("2.0.0"))
            and compile_model
            else lambda x: x
        )

        self.encoder = compile(instantiate_from_config(encoder_config))
        self.decoder = compile(instantiate_from_config(decoder_config))
        self.loss = instantiate_from_config(loss_config)
        self.regularization = instantiate_from_config(regularizer_config)
        
        # define the qformer
        self.temporal_qformer = instantiate_from_config(temporal_qformer_config)
        self.space_qformer = instantiate_from_config(space_qformer_config)
        
        
        self.partial_content_motion = partial_content_motion
        self.shuffle_content = shuffle_content
        self.repeat_for_decoder = repeat_for_decoder
        
        self.use_scheduler = lr_scheduler_config is not None
        self.check = 0
        self.weight_decay = weight_decay
        if self.use_scheduler:
            self.lr_scheduler_config = lr_scheduler_config
        self.use_scheduler_g = lr_scheduler_config_g is not None 
        self.use_scheduler_d = lr_scheduler_config_d is not None
        if self.use_scheduler_g:
            self.lr_scheduler_config_g = lr_scheduler_config_g
        if self.use_scheduler_d:
            self.lr_scheduler_config_d = lr_scheduler_config_d
        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.Adam", "params": {"betas": (0, 0.99), "weight_decay": self.weight_decay}})
        self.trainable_ae_params = trainable_ae_params
        if self.trainable_ae_params is not None:
            self.ae_optimizer_args = default(
                ae_optimizer_args,
                [{} for _ in range(len(self.trainable_ae_params))],
            )
            assert len(self.ae_optimizer_args) == len(self.trainable_ae_params)
        else:
            self.ae_optimizer_args = [{}]  # makes type consitent
        self.trainable_disc_params = trainable_disc_params
        if self.trainable_disc_params is not None:
            self.disc_optimizer_args = default(
                disc_optimizer_args,
                [{} for _ in range(len(self.trainable_disc_params))],
            )
            assert len(self.disc_optimizer_args) == len(self.trainable_disc_params)
        else:
            self.disc_optimizer_args = [{}]  # makes type consitent
        
        self.lr_g_factor = lr_g_factor

        self.hidden_dim = encoder_config.params.hidden_size
        self.patch_nums = np.array(list(encoder_config.params.input_size)) // np.array(list(encoder_config.params.patch_size))
        
        self.temporal_down_dim = temporal_down_dim
        self.down_channel_temp = nn.Linear(self.hidden_dim, self.temporal_down_dim)
        self.up_channel_temp = nn.Linear(self.temporal_down_dim, self.hidden_dim)
        self.pre_temporal_qformer = nn.Sequential(
            nn.Linear(self.temporal_down_dim * self.patch_nums[1] * self.patch_nums[2], self.hidden_dim),
            nn.ReLU(),
        )
        self.retain_num_frames = retain_num_frames
        if not self.retain_num_frames:
            self.pre_spatial_qformer = nn.Sequential(
                nn.Linear(self.hidden_dim * self.patch_nums[0], 2 * self.hidden_dim),
                nn.ReLU(),
                nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
            )
        if self.repeat_for_decoder:
            self.cont_emb = nn.Sequential(
                nn.Linear(temporal_qformer_config.params.query_hidden_size, self.hidden_dim),
                nn.ReLU(),
                nn.Conv1d(temporal_qformer_config.params.num_query_tokens, self.patch_nums[1] * self.patch_nums[2], 1),
                nn.ReLU(),
            )
        else:    
            # (bhw, f, c) -> (bhw, f',c')
            self.cont_emb = nn.Sequential(
                nn.Linear(temporal_qformer_config.params.query_hidden_size, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.temporal_down_dim * self.patch_nums[1] * self.patch_nums[2]),
                nn.ReLU(),
                nn.Conv1d(temporal_qformer_config.params.num_query_tokens, self.patch_nums[0], 1),
                nn.ReLU(),
            )
        
        if retain_num_frames:
            self.spatial_emb = nn.Sequential(
                nn.Linear(space_qformer_config.params.query_hidden_size, self.hidden_dim),
                nn.ReLU(),
                nn.Conv1d(space_qformer_config.params.num_query_tokens, self.patch_nums[1] * self.patch_nums[2], 1),
                nn.ReLU(),
            )
        else:
            self.spatial_emb = nn.Sequential(
                nn.Linear(space_qformer_config.params.query_hidden_size, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim * self.patch_nums[0]),
                nn.ReLU(),
                nn.Conv1d(space_qformer_config.params.num_query_tokens, self.patch_nums[1] * self.patch_nums[2], 1),
                nn.ReLU(),
            )
        
        ckpt_path = get_valid_paths(ckpt_path, ckpt_path2)
        print0(f"[bold magenta]\[vidtok.models.vidtwin_ae][VidAutoencoderQformer][/bold magenta] Use ckpt_path: {ckpt_path}")
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
            
    def get_disentangle_params(self) -> list:
        params = (
            list(self.temporal_qformer.parameters())
            + list(self.space_qformer.parameters())
            + list(self.cont_emb.parameters())
            + list(self.spatial_emb.parameters())
            + list(self.pre_temporal_qformer.parameters())
            + list(self.down_channel_temp.parameters())
        )
        if not self.retain_num_frames:
            params += list(self.pre_spatial_qformer.parameters())
        if not self.repeat_for_decoder:
            params += list(self.up_channel_temp.parameters())    
        return params
    
    def decode(self, z,  z_content, z_motion, only_part=None) -> torch.Tensor:
        '''
        input: z: shape (b, c', f, h', w')
                z_content: shape (b, f_q, c_q)
                z_motion: shape (b, [f] , s_q, c_q)
        '''
        if self.repeat_for_decoder:
            z_content = repeat(z_content, 'B F C -> B f F C', f=z.size(2))
            vt = rearrange(self.cont_emb(rearrange(z_content, 'B F A d -> (B F) A d')), '(B f) (H W) C -> B C f H W', H=z.size(3), W=z.size(4), f=z.size(2))
        else:
            vt = rearrange(self.cont_emb(z_content), 'B F (C H W) -> B C F H W', H=z.size(3), W=z.size(4))
            vt = self.up_channel_temp(vt.transpose(1, -1)).transpose(1, -1)
        if self.retain_num_frames:
            vs = rearrange(self.spatial_emb(rearrange(z_motion, 'B F X Y -> (B F) X Y')), '(B F) (H W) C -> B C F H W', H=z.size(3), W=z.size(4), F=z.size(2))
        else:
            vs = rearrange(self.spatial_emb(z_motion), 'B (H W) (F C) -> B C F H W', H=z.size(3), W=z.size(4), F=z.size(2))
            
        if self.partial_content_motion == 'content':
            c_plus_m = vt
        elif self.partial_content_motion == 'motion':
            c_plus_m = vs
        else:
            c_plus_m = vt + vs # shape (b, c', f, h', w')
        if only_part == 'content':
            c_plus_m = vt
        elif only_part == 'motion':
            c_plus_m = vs    
        x = self.decoder(c_plus_m)
        return x

    def encode(self, x: Any, return_reg_log: bool = False) -> Any:
        z = self.encoder(x) # shape (b, c', f, h', w')
        if self.shuffle_content:
            b, c, f, h, w = z.shape
            z_shuffled = torch.empty_like(z)
            for i in range(b):
                idx = torch.randperm(f)
                z_shuffled[i] = z[i, :, idx, :, :]
            pre_qformer = self.pre_temporal_qformer(rearrange(self.down_channel_temp(rearrange(z_shuffled, 'B C F H W -> B F H W C')), 'B F H W C -> B F (H W C)'))   
        else: 
            pre_qformer = self.pre_temporal_qformer(rearrange(self.down_channel_temp(rearrange(z, 'B C F H W -> B F H W C')), 'B F H W C -> B F (H W C)'))
        z_content = self.temporal_qformer(pre_qformer) # shape (b, f_q, d_q)
        layer_norm_content = nn.LayerNorm(z_content.size(-1)).to(z_content.device)
        z_content = layer_norm_content(z_content)
        
        # intuitively, we can view the z_content as a method to retrieve the content frames (including its nums and dims)
        if self.retain_num_frames:
            z_motion = self.space_qformer(rearrange(z, 'B C F H W -> (B F) (H W) C')) # shape (bf, n_q, d_q)
            # for each frame, we use qformer to compress the spatial dimension
            z_motion = rearrange(z_motion, '(B F) a b -> B F a b', F=z.size(2))
        else:
            z_motion = self.space_qformer(self.pre_spatial_qformer(rearrange(z, 'B C F H W -> B (H W) (F C)'))) 
        layer_norm_motion = nn.LayerNorm(z_motion.size(-1)).to(z_motion.device)
        z_motion = layer_norm_motion(z_motion)
        if return_reg_log:
            return z, z_content, z_motion, None
        return z, z_content, z_motion

    
    def forward(self, x: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (bs, 3, 17, h, w)
        z, z_content, z_motion, reg_log = self.encode(x, return_reg_log=True)
        # z: shape (b, c', f, h', w')
        dec = self.decode(z, z_content, z_motion)
        # dec: (bs, 3, 17, h, w)
        return z, dec, reg_log, z_content, z_motion


class VidAutoEncoderQformerCompactSym(VidAutoEncoderQformerBase):
    
    def __init__(
        self,
        *args,
        encoder_config: Dict,
        decoder_config: Dict,
        loss_config: Dict,
        regularizer_config: Dict,
        temporal_qformer_config: Dict,
        space_qformer_config: Dict,
        lr_scheduler_config_g=None,
        lr_scheduler_config_d=None,
        trainable_ae_params=None,
        ae_optimizer_args = None,
        trainable_disc_params = None,
        lr_scheduler_config: Dict = None,
        weight_decay: float = 1e-5,
        disc_optimizer_args = None,
        optimizer_config: Union[Dict, None] = None,
        lr_g_factor: float = 1.0,
        compile_model: bool = False,
        retain_num_frames: bool = True,
        temporal_down_dim: int = 32,
        partial_content_motion: str = 'all',
        shuffle_content: bool = False,
        init_ch: int = 128,
        cont_num_blocks: int = 2,
        expect_ch: int = 4,
        **kwargs,
    ):
        ckpt_path = kwargs.pop("ckpt_path", None)
        ckpt_path2 = kwargs.pop("ckpt_path2", None)
        ignore_keys = kwargs.pop("ignore_keys", ())
        super().__init__(*args, **kwargs)
        compile = (
            torch.compile
            if (version.parse(torch.__version__) >= version.parse("2.0.0"))
            and compile_model
            else lambda x: x
        )

        self.encoder = compile(instantiate_from_config(encoder_config))
        self.decoder = compile(instantiate_from_config(decoder_config))
        self.loss = instantiate_from_config(loss_config)
        self.regularization = instantiate_from_config(regularizer_config)
        
        # define the qformer
        self.temporal_qformer = instantiate_from_config(temporal_qformer_config)
        self.space_qformer = instantiate_from_config(space_qformer_config)
        
        
        self.partial_content_motion = partial_content_motion
        self.shuffle_content = shuffle_content
        
        self.use_scheduler = lr_scheduler_config is not None
        self.check = 0
        self.weight_decay = weight_decay
        if self.use_scheduler:
            self.lr_scheduler_config = lr_scheduler_config
        self.use_scheduler_g = lr_scheduler_config_g is not None 
        self.use_scheduler_d = lr_scheduler_config_d is not None
        if self.use_scheduler_g:
            self.lr_scheduler_config_g = lr_scheduler_config_g
        if self.use_scheduler_d:
            self.lr_scheduler_config_d = lr_scheduler_config_d
        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.Adam", "params": {"betas": (0, 0.99), "weight_decay": self.weight_decay}})
        self.trainable_ae_params = trainable_ae_params
        if self.trainable_ae_params is not None:
            self.ae_optimizer_args = default(
                ae_optimizer_args,
                [{} for _ in range(len(self.trainable_ae_params))],
            )
            assert len(self.ae_optimizer_args) == len(self.trainable_ae_params)
        else:
            self.ae_optimizer_args = [{}]  # makes type consitent
        self.trainable_disc_params = trainable_disc_params
        if self.trainable_disc_params is not None:
            self.disc_optimizer_args = default(
                disc_optimizer_args,
                [{} for _ in range(len(self.trainable_disc_params))],
            )
            assert len(self.disc_optimizer_args) == len(self.trainable_disc_params)
        else:
            self.disc_optimizer_args = [{}]  # makes type consitent
        
        self.lr_g_factor = lr_g_factor

        self.hidden_dim = encoder_config.params.hidden_size
        self.patch_nums = np.array(list(encoder_config.params.input_size)) // np.array(list(encoder_config.params.patch_size))
        
        self.temporal_down_dim = temporal_down_dim
        self.retain_num_frames = retain_num_frames
        if not self.retain_num_frames:
            self.pre_spatial_qformer = nn.Sequential(
                nn.Linear(self.hidden_dim * self.patch_nums[0], 2 * self.hidden_dim),
                nn.ReLU(),
                nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
            )

        self.cont_emb = nn.Sequential(
            nn.Linear(temporal_qformer_config.params.query_hidden_size, self.hidden_dim),
            nn.ReLU(),
            nn.Conv1d(temporal_qformer_config.params.num_query_tokens, self.patch_nums[0], 1),
            nn.ReLU(),
        )
        
        if retain_num_frames:
            self.spatial_emb = nn.Sequential(
                nn.Linear(space_qformer_config.params.query_hidden_size, self.hidden_dim),
                nn.ReLU(),
                nn.Conv1d(space_qformer_config.params.num_query_tokens, self.patch_nums[1] * self.patch_nums[2], 1),
                nn.ReLU(),
            )
        else:
            self.spatial_emb = nn.Sequential(
                nn.Linear(space_qformer_config.params.query_hidden_size, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim * self.patch_nums[0]),
                nn.ReLU(),
                nn.Conv1d(space_qformer_config.params.num_query_tokens, self.patch_nums[1] * self.patch_nums[2], 1),
                nn.ReLU(),
            )
        
        
        downsample_blocks = []
        in_channels = temporal_qformer_config.params.query_hidden_size
        self.init_ch = init_ch
        self.conv_in = nn.Conv2d(in_channels, self.init_ch, kernel_size=3, stride=1, padding=1)
        in_channels = self.init_ch
            

        for i in range(cont_num_blocks):
            out_channels = 2 * in_channels
            downsample_blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
            downsample_blocks.append(nn.ReLU())
            in_channels = out_channels
        self.content_downsample_blocks = nn.Sequential(*downsample_blocks)
        
        self.max_channels = in_channels    
        upsample_blocks = []
        for i in range(cont_num_blocks):
            out_channels = in_channels // 2
            upsample_blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            upsample_blocks.append(nn.ReLU())
            upsample_blocks.append(nn.Upsample(scale_factor=2))
            in_channels = out_channels
        self.content_upsample_blocks = nn.Sequential(*upsample_blocks)
        

        self.bottle_down = nn.Conv2d(self.max_channels, expect_ch, kernel_size=3, stride=1, padding=1)
        self.bottle_up = nn.Sequential(
                                nn.Conv2d(expect_ch, self.max_channels, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU())
        self.conv_out = nn.Conv2d(self.init_ch, temporal_qformer_config.params.query_hidden_size, kernel_size=3, stride=1, padding=1)
        

        
        
        ckpt_path = get_valid_paths(ckpt_path, ckpt_path2)
        print0(f"[bold magenta]\[vidtok.models.vidtwin_ae][VidAutoencoderQformer][/bold magenta] Use ckpt_path: {ckpt_path}")
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
            
    def get_disentangle_params(self) -> list:
        params = (
            list(self.temporal_qformer.parameters())
            + list(self.space_qformer.parameters())
            + list(self.cont_emb.parameters())
            + list(self.spatial_emb.parameters())
            + list(self.conv_in.parameters())
            + list(self.content_downsample_blocks.parameters())
            + list(self.bottle_down.parameters())
            + list(self.bottle_up.parameters())
            + list(self.conv_out.parameters())
            + list(self.content_upsample_blocks.parameters())
            
        )
        if not self.retain_num_frames:
            params += list(self.pre_spatial_qformer.parameters())
            
        return params
    
    def decode(self, z,  z_content, z_motion) -> torch.Tensor:
        '''
        input: z: shape (b, c', f, h', w')
                z_content: shape (b, f_q, h_q, w_q, c_q)
                z_motion: shape (b, [f] , s_q, c_q)
        '''

        z_content_up = self.conv_out(self.content_upsample_blocks(self.bottle_up(rearrange(z_content, 'B F H W C -> (B F) C H W'))))
        _,_,h,w = z_content_up.shape
        if h > z.size(3):
            border = (h - z.size(3)) // 2
            z_content_up = z_content_up[:, :, border:border+z.size(3), border:border+z.size(4)]
        z_content = rearrange(z_content_up, '(B F) C H W -> (B H W) F C', F=z_content.size(1))
        vt = rearrange(self.cont_emb(z_content), '(B H W) F C -> B C F H W', H=z.size(3), W=z.size(4))
        
        if self.retain_num_frames:
            vs = rearrange(self.spatial_emb(rearrange(z_motion, 'B F X Y -> (B F) X Y')), '(B F) (H W) C -> B C F H W', H=z.size(3), W=z.size(4), F=z.size(2))
        else:
            vs = rearrange(self.spatial_emb(z_motion), 'B (H W) (F C) -> B C F H W', H=z.size(3), W=z.size(4), F=z.size(2))
            
        if self.partial_content_motion == 'content':
            c_plus_m = vt
        elif self.partial_content_motion == 'motion':
            c_plus_m = vs
        else:
            c_plus_m = vt + vs # shape (b, c', f, h', w')
            
        x = self.decoder(c_plus_m)
        return x

    def encode(self, x: Any, return_reg_log: bool = False) -> Any:
        z = self.encoder(x) # shape (b, c', f, h', w')
        if self.shuffle_content:
            b, c, f, h, w = z.shape
            z_shuffled = torch.empty_like(z)
            for i in range(b):
                idx = torch.randperm(f)
                z_shuffled[i] = z[i, :, idx, :, :]
            pre_qformer = rearrange(z_shuffled, 'B C F H W -> (B H W) F C')   
        else: 
            pre_qformer = rearrange(z, 'B C F H W -> (B H W) F C')
        z_content = self.temporal_qformer(pre_qformer) # shape (bhw, f_q, d_q)
        z_content_down = self.bottle_down(self.content_downsample_blocks(self.conv_in(rearrange(z_content, '(B H W) F C -> (B F) C H W', H=z.size(3), W=z.size(4)))))
        z_content = rearrange(z_content_down, '(B F) C H W -> B F H W C', F=z_content.size(1))
        # intuitively, we can view the z_content as a method to retrieve the content frames (including its nums and dims)
        if self.retain_num_frames:
            z_motion = self.space_qformer(rearrange(z, 'B C F H W -> (B F) (H W) C')) # shape (bf, n_q, d_q)
            # for each frame, we use qformer to compress the spatial dimension
            z_motion = rearrange(z_motion, '(B F) a b -> B F a b', F=z.size(2))
        else:
            z_motion = self.space_qformer(self.pre_spatial_qformer(rearrange(z, 'B C F H W -> B (H W) (F C)'))) 
        if return_reg_log:
            # return z, z_content, z_motion_x, z_motion_y, reg_log
            return z, z_content, z_motion, None
        return z, z_content, z_motion

    
    def forward(self, x: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (bs, 3, 17, h, w)
        z, z_content, z_motion, reg_log = self.encode(x, return_reg_log=True)
        # z: shape (b, c', f, h', w')
        dec = self.decode(z, z_content, z_motion)
        # dec: (bs, 3, 17, h, w)
        return z, dec, reg_log, z_content, z_motion


class VidAutoEncoderQformerCompactSymDis(VidAutoEncoderQformerCompactSym):
    
    def __init__(
        self,
        *args,
        shuffle_content_ratio: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.shuffle_content_ratio = shuffle_content_ratio
            

    def encode(self, x: Any, return_reg_log: bool = False) -> Any:
        # shuffle the content frames
        x_shuffled = x.clone()
        for i in range(x.size(0)):
            randn_num = torch.rand(1)
            if randn_num < self.shuffle_content_ratio:
                idx = torch.randperm(x.size(2))
                x_shuffled[i] = x[i, :, idx, :, :]
        x = torch.cat([x, x_shuffled], dim=0)
        z = self.encoder(x) # shape (2b, c', f, h', w')
        z_orig, z_shuffled = z.chunk(2, dim=0)
        pre_qformer = rearrange(z_shuffled, 'B C F H W -> (B H W) F C')
        z_content = self.temporal_qformer(pre_qformer) # shape (bhw, f_q, d_q)
        z_content_down = self.bottle_down(self.content_downsample_blocks(self.conv_in(rearrange(z_content, '(B H W) F C -> (B F) C H W', H=z.size(3), W=z.size(4)))))
        z_content = rearrange(z_content_down, '(B F) C H W -> B F H W C', F=z_content.size(1))
        # intuitively, we can view the z_content as a method to retrieve the content frames (including its nums and dims)
        if self.retain_num_frames:
            z_motion = self.space_qformer(rearrange(z_orig, 'B C F H W -> (B F) (H W) C')) # shape (bf, n_q, d_q)
            # for each frame, we use qformer to compress the spatial dimension
            z_motion = rearrange(z_motion, '(B F) a b -> B F a b', F=z.size(2))
        else:
            z_motion = self.space_qformer(self.pre_spatial_qformer(rearrange(z_orig, 'B C F H W -> B (H W) (F C)'))) 
        if return_reg_log:
            # return z, z_content, z_motion_x, z_motion_y, reg_log
            return z, z_content, z_motion, None
        return z, z_content, z_motion

class VidAutoEncoderQformerCompactSymVid(VidAutoEncoderQformerBase):
    
    def __init__(
        self,
        *args,
        encoder_config: Dict,
        decoder_config: Dict,
        loss_config: Dict,
        regularizer_config: Dict,
        temporal_qformer_config: Dict,
        lr_scheduler_config_g=None,
        lr_scheduler_config_d=None,
        trainable_ae_params=None,
        ae_optimizer_args = None,
        trainable_disc_params = None,
        lr_scheduler_config: Dict = None,
        weight_decay: float = 1e-5,
        disc_optimizer_args = None,
        optimizer_config: Union[Dict, None] = None,
        lr_g_factor: float = 1.0,
        compile_model: bool = False,
        temporal_down_dim: int = 32,
        partial_content_motion: str = 'all',
        shuffle_content: bool = False,
        init_ch: int = 128,
        cont_num_blocks: int = 2,
        motion_num_blocks: int = 2,
        expect_ch: int = 4,
        d_dim: int = 16,
        # space_qformer_config: Dict,
        downsample_motion: bool = False,
        **kwargs,
    ):
        ckpt_path = kwargs.pop("ckpt_path", None)
        ckpt_path2 = kwargs.pop("ckpt_path2", None)
        ignore_keys = kwargs.pop("ignore_keys", ())
        super().__init__(*args, **kwargs)
        compile = (
            torch.compile
            if (version.parse(torch.__version__) >= version.parse("2.0.0"))
            and compile_model
            else lambda x: x
        )

        self.encoder = compile(instantiate_from_config(encoder_config))
        self.decoder = compile(instantiate_from_config(decoder_config))
        self.loss = instantiate_from_config(loss_config)
        self.regularization = instantiate_from_config(regularizer_config)
        
        # define the qformer
        self.temporal_qformer = instantiate_from_config(temporal_qformer_config)        
        
        self.partial_content_motion = partial_content_motion
        self.shuffle_content = shuffle_content
        
        self.use_scheduler = lr_scheduler_config is not None
        self.check = 0
        self.weight_decay = weight_decay
        if self.use_scheduler:
            self.lr_scheduler_config = lr_scheduler_config
        self.use_scheduler_g = lr_scheduler_config_g is not None 
        self.use_scheduler_d = lr_scheduler_config_d is not None
        if self.use_scheduler_g:
            self.lr_scheduler_config_g = lr_scheduler_config_g
        if self.use_scheduler_d:
            self.lr_scheduler_config_d = lr_scheduler_config_d
        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.Adam", "params": {"betas": (0, 0.99), "weight_decay": self.weight_decay}})
        self.trainable_ae_params = trainable_ae_params
        if self.trainable_ae_params is not None:
            self.ae_optimizer_args = default(
                ae_optimizer_args,
                [{} for _ in range(len(self.trainable_ae_params))],
            )
            assert len(self.ae_optimizer_args) == len(self.trainable_ae_params)
        else:
            self.ae_optimizer_args = [{}]  # makes type consitent
        self.trainable_disc_params = trainable_disc_params
        if self.trainable_disc_params is not None:
            self.disc_optimizer_args = default(
                disc_optimizer_args,
                [{} for _ in range(len(self.trainable_disc_params))],
            )
            assert len(self.disc_optimizer_args) == len(self.trainable_disc_params)
        else:
            self.disc_optimizer_args = [{}]  # makes type consitent
        
        self.lr_g_factor = lr_g_factor

        self.hidden_dim = encoder_config.params.hidden_size
        self.patch_nums = np.array(list(encoder_config.params.input_size)) // np.array(list(encoder_config.params.patch_size))
        
        self.temporal_down_dim = temporal_down_dim

        self.cont_emb = nn.Sequential(
            nn.Linear(temporal_qformer_config.params.query_hidden_size, self.hidden_dim),
            nn.ReLU(),
            nn.Conv1d(temporal_qformer_config.params.num_query_tokens, self.patch_nums[0], 1),
            nn.ReLU(),
        )
        
        self.d_dim = d_dim
        
        
        downsample_blocks = []
        in_channels = temporal_qformer_config.params.query_hidden_size
        self.init_ch = init_ch
        self.conv_in = nn.Conv2d(in_channels, self.init_ch, kernel_size=3, stride=1, padding=1)
        in_channels = self.init_ch
            

        for i in range(cont_num_blocks):
            out_channels = 2 * in_channels
            downsample_blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
            downsample_blocks.append(nn.ReLU())
            in_channels = out_channels
        self.content_downsample_blocks = nn.Sequential(*downsample_blocks)
        
        self.max_channels = in_channels    
        upsample_blocks = []
        for i in range(cont_num_blocks):
            out_channels = in_channels // 2
            upsample_blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            upsample_blocks.append(nn.ReLU())
            upsample_blocks.append(nn.Upsample(scale_factor=2))
            in_channels = out_channels
        self.content_upsample_blocks = nn.Sequential(*upsample_blocks)
        

        self.bottle_down = nn.Conv2d(self.max_channels, expect_ch, kernel_size=3, stride=1, padding=1)
        self.bottle_up = nn.Sequential(
                                nn.Conv2d(expect_ch, self.max_channels, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU())
        self.conv_out = nn.Conv2d(self.init_ch, temporal_qformer_config.params.query_hidden_size, kernel_size=3, stride=1, padding=1)
        
        self.motion_emb = nn.Sequential(
            nn.Linear(self.d_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )
        self.motion_head = nn.Conv2d(self.hidden_dim, self.d_dim, kernel_size=3, stride=1, padding=1)
        
        self.downsample_motion = downsample_motion
        if self.downsample_motion:
            motion_downsample_blocks = []
            curr_resol = self.patch_nums[1]
            for i in range(motion_num_blocks):
                motion_downsample_blocks.append(nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=2, padding=1))
                motion_downsample_blocks.append(nn.ReLU())
                curr_resol = (curr_resol + 1) // 2
            self.downsample_motion_module = nn.Sequential(*motion_downsample_blocks)
            self.up_motion = nn.Sequential(nn.Linear(curr_resol, self.patch_nums[1]), 
                                           nn.ReLU(),
                                           nn.Linear(self.patch_nums[1], self.patch_nums[1]),
                                           nn.ReLU())
        
        
        ckpt_path = get_valid_paths(ckpt_path, ckpt_path2)
        print0(f"[bold magenta]\[vidtok.models.vidtwin_ae][VidAutoencoderQformer][/bold magenta] Use ckpt_path: {ckpt_path}")
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
            
    def get_disentangle_params(self) -> list:
        params = (
            list(self.temporal_qformer.parameters())
            + list(self.cont_emb.parameters())
            + list(self.conv_in.parameters())
            + list(self.content_downsample_blocks.parameters())
            + list(self.bottle_down.parameters())
            + list(self.bottle_up.parameters())
            + list(self.conv_out.parameters())
            + list(self.content_upsample_blocks.parameters())
            + list(self.motion_emb.parameters())
            + list(self.motion_head.parameters())
            
        )
        if self.downsample_motion:
            params += list(self.downsample_motion_module.parameters())
            params += list(self.up_motion.parameters())
            
        return params
    
    def decode(self, z,  z_content, z_motion_x, z_motion_y) -> torch.Tensor:
        '''
        input: z: shape (b, c', f, h', w')
                z_content: shape (b, f_q, h_q, w_q, c_q)
                z_motion: shape (b, [f] , s_q, c_q)
        '''

        z_content_up = self.conv_out(self.content_upsample_blocks(self.bottle_up(rearrange(z_content, 'B F H W C -> (B F) C H W'))))
        _,_,h,w = z_content_up.shape
        if h > z.size(3):
            border = (h - z.size(3)) // 2
            z_content_up = z_content_up[:, :, border:border+z.size(3), border:border+z.size(4)]
        z_content = rearrange(z_content_up, '(B F) C H W -> (B H W) F C', F=z_content.size(1))
        vt = rearrange(self.cont_emb(z_content), '(B H W) F C -> B C F H W', H=z.size(3), W=z.size(4))
        
        vx = rearrange(self.motion_emb(rearrange(z_motion_x, 'B D F W -> B F W D')), 'B F W C -> B C F W') # shape (b, c', f, w')
        vy = rearrange(self.motion_emb(rearrange(z_motion_y, 'B D F H -> B F H D')), 'B F H C -> B C F H') # shape (b, c', f, h')
        if self.downsample_motion:
            vx = self.up_motion(vx)
            vy = self.up_motion(vy)
        vx = repeat(vx, 'b c f w -> b c f h w', h=z.size(3))
        vy = repeat(vy, 'b c f h -> b c f h w', w=z.size(4))
            
        c_plus_m = vt + vx + vy # shape (b, c', f, h', w')
            
        x = self.decoder(c_plus_m)
        return x

    def encode(self, x: Any, return_reg_log: bool = False) -> Any:
        z = self.encoder(x) # shape (b, c', f, h', w')
        if self.shuffle_content:
            b, c, f, h, w = z.shape
            z_shuffled = torch.empty_like(z)
            for i in range(b):
                idx = torch.randperm(f)
                z_shuffled[i] = z[i, :, idx, :, :]
            pre_qformer = rearrange(z_shuffled, 'B C F H W -> (B H W) F C')   
        else: 
            pre_qformer = rearrange(z, 'B C F H W -> (B H W) F C')
        z_content = self.temporal_qformer(pre_qformer) # shape (bhw, f_q, d_q)
        z_content_down = self.bottle_down(self.content_downsample_blocks(self.conv_in(rearrange(z_content, '(B H W) F C -> (B F) C H W', H=z.size(3), W=z.size(4)))))
        z_content = rearrange(z_content_down, '(B F) C H W -> B F H W C', F=z_content.size(1))
        # intuitively, we can view the z_content as a method to retrieve the content frames (including its nums and dims)
        z_motion_x, z_motion_y = self.get_motion_latent(z)
        
        if return_reg_log:
            return z, z_content, z_motion_x, z_motion_y, None
        return z, z_content, z_motion_x, z_motion_y

    def get_motion_latent(self, z: torch.Tensor) -> torch.Tensor:
        f = z.size(2)
        if self.downsample_motion:
            z = self.downsample_motion_module(rearrange(z, 'B C F H W -> (B F) C H W'))
            z = rearrange(z, '(B F) C H W -> B C F H W', F=f)
        ux = torch.mean(z, dim=-2) # shape (b, c', f, w')
        uy = torch.mean(z, dim=-1) # shape (b, c', f, h')
        zx = self.motion_head(ux) # shape (b, d, f, w')
        zy = self.motion_head(uy) # shape (b, d, f, h')      
        return zx, zy
    
    def forward(self, x: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (bs, 3, 17, h, w)
        z, z_content, z_motion_x, z_motion_y, reg_log = self.encode(x, return_reg_log=True)
        # z: shape (b, c', f, h', w')
        dec = self.decode(z, z_content, z_motion_x, z_motion_y)
        # dec: (bs, 3, 17, h, w)
        return z, dec, reg_log, z_content, z_motion_x, z_motion_y



class VidAutoEncoderQformerCompactSymVidVAE(VidAutoEncoderQformerBase):
    
    def __init__(
        self,
        *args,
        encoder_config: Dict,
        decoder_config: Dict,
        loss_config: Dict,
        regularizer_config: Dict,
        temporal_qformer_config: Dict,
        lr_scheduler_config_g=None,
        lr_scheduler_config_d=None,
        trainable_ae_params=None,
        ae_optimizer_args = None,
        trainable_disc_params = None,
        lr_scheduler_config: Dict = None,
        weight_decay: float = 1e-5,
        disc_optimizer_args = None,
        optimizer_config: Union[Dict, None] = None,
        lr_g_factor: float = 1.0,
        compile_model: bool = False,
        temporal_down_dim: int = 32,
        partial_content_motion: str = 'all',
        shuffle_content: bool = False,
        init_ch: int = 128,
        cont_num_blocks: int = 2,
        motion_num_blocks: int = 2,
        expect_ch: int = 4,
        d_dim: int = 16,
        downsample_motion: bool = False,
        **kwargs,
    ):
        ckpt_path = kwargs.pop("ckpt_path", None)
        ckpt_path2 = kwargs.pop("ckpt_path2", None)
        ignore_keys = kwargs.pop("ignore_keys", ())
        super().__init__(*args, **kwargs)
        compile = (
            torch.compile
            if (version.parse(torch.__version__) >= version.parse("2.0.0"))
            and compile_model
            else lambda x: x
        )

        self.encoder = compile(instantiate_from_config(encoder_config))
        self.decoder = compile(instantiate_from_config(decoder_config))
        self.loss = instantiate_from_config(loss_config)
        self.regularization = instantiate_from_config(regularizer_config)
        
        # define the qformer
        self.temporal_qformer = instantiate_from_config(temporal_qformer_config)        
        
        self.partial_content_motion = partial_content_motion
        self.shuffle_content = shuffle_content
        
        self.use_scheduler = lr_scheduler_config is not None
        self.check = 0
        self.weight_decay = weight_decay
        if self.use_scheduler:
            self.lr_scheduler_config = lr_scheduler_config
        self.use_scheduler_g = lr_scheduler_config_g is not None 
        self.use_scheduler_d = lr_scheduler_config_d is not None
        if self.use_scheduler_g:
            self.lr_scheduler_config_g = lr_scheduler_config_g
        if self.use_scheduler_d:
            self.lr_scheduler_config_d = lr_scheduler_config_d
        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.Adam", "params": {"betas": (0, 0.99), "weight_decay": self.weight_decay}})
        self.trainable_ae_params = trainable_ae_params
        if self.trainable_ae_params is not None:
            self.ae_optimizer_args = default(
                ae_optimizer_args,
                [{} for _ in range(len(self.trainable_ae_params))],
            )
            assert len(self.ae_optimizer_args) == len(self.trainable_ae_params)
        else:
            self.ae_optimizer_args = [{}]  # makes type consitent
        self.trainable_disc_params = trainable_disc_params
        if self.trainable_disc_params is not None:
            self.disc_optimizer_args = default(
                disc_optimizer_args,
                [{} for _ in range(len(self.trainable_disc_params))],
            )
            assert len(self.disc_optimizer_args) == len(self.trainable_disc_params)
        else:
            self.disc_optimizer_args = [{}]  # makes type consitent
        
        self.lr_g_factor = lr_g_factor

        self.hidden_dim = encoder_config.params.hidden_size
        self.patch_nums = np.array(list(encoder_config.params.input_size)) // np.array(list(encoder_config.params.patch_size))
        
        self.temporal_down_dim = temporal_down_dim
        self.cont_emb = nn.Sequential(
            nn.Linear(temporal_qformer_config.params.query_hidden_size, self.hidden_dim),
            nn.ReLU(),
            nn.Conv1d(temporal_qformer_config.params.num_query_tokens, self.patch_nums[0], 1),
            nn.ReLU(),
        )
        
        self.d_dim = d_dim
        
        
        downsample_blocks = []
        in_channels = temporal_qformer_config.params.query_hidden_size
        self.init_ch = init_ch
        self.conv_in = nn.Conv2d(in_channels, self.init_ch, kernel_size=3, stride=1, padding=1)
        in_channels = self.init_ch
            

        for i in range(cont_num_blocks):
            out_channels = 2 * in_channels
            downsample_blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
            downsample_blocks.append(nn.ReLU())
            in_channels = out_channels
        self.content_downsample_blocks = nn.Sequential(*downsample_blocks)
        
        self.max_channels = in_channels    
        upsample_blocks = []
        for i in range(cont_num_blocks):
            out_channels = in_channels // 2
            upsample_blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            upsample_blocks.append(nn.ReLU())
            upsample_blocks.append(nn.Upsample(scale_factor=2))
            in_channels = out_channels
        self.content_upsample_blocks = nn.Sequential(*upsample_blocks)
        

        self.bottle_down = nn.Conv2d(self.max_channels, 2*expect_ch, kernel_size=3, stride=1, padding=1)
        self.bottle_up = nn.Sequential(
                                nn.Conv2d(expect_ch, self.max_channels, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU())
        self.conv_out = nn.Conv2d(self.init_ch, temporal_qformer_config.params.query_hidden_size, kernel_size=3, stride=1, padding=1)
        
        self.motion_emb = nn.Sequential(
            nn.Linear(self.d_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )
        self.motion_head = nn.Conv2d(self.hidden_dim, 2*self.d_dim, kernel_size=3, stride=1, padding=1)
        
        self.downsample_motion = downsample_motion
        if self.downsample_motion:
            motion_downsample_blocks = []
            curr_resol = self.patch_nums[1]
            for i in range(motion_num_blocks):
                motion_downsample_blocks.append(nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=2, padding=1))
                motion_downsample_blocks.append(nn.ReLU())
                curr_resol = (curr_resol + 1) // 2
            self.downsample_motion_module = nn.Sequential(*motion_downsample_blocks)
            self.up_motion = nn.Sequential(nn.Linear(curr_resol, self.patch_nums[1]), 
                                           nn.ReLU(),
                                           nn.Linear(self.patch_nums[1], self.patch_nums[1]),
                                           nn.ReLU())
        
        
        ckpt_path = get_valid_paths(ckpt_path, ckpt_path2)
        print0(f"[bold magenta]\[vidtok.models.vidtwin_ae][VidAutoencoderQformer][/bold magenta] Use ckpt_path: {ckpt_path}")
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
            
    def get_disentangle_params(self) -> list:
        params = (
            list(self.temporal_qformer.parameters())
            + list(self.cont_emb.parameters())
            + list(self.conv_in.parameters())
            + list(self.content_downsample_blocks.parameters())
            + list(self.bottle_down.parameters())
            + list(self.bottle_up.parameters())
            + list(self.conv_out.parameters())
            + list(self.content_upsample_blocks.parameters())
            + list(self.motion_emb.parameters())
            + list(self.motion_head.parameters())
            
        )
        if self.downsample_motion:
            params += list(self.downsample_motion_module.parameters())
            params += list(self.up_motion.parameters())
            
        return params
            
    
    def decode(self, z,  z_content, z_motion_x, z_motion_y, only_part=None) -> torch.Tensor:
        '''
        input: z: shape (b, c', f, h', w')
                z_content: shape (b, f_q, h_q, w_q, c_q)
                z_motion: shape (b, [f] , s_q, c_q)
        '''

        z_content_up = self.conv_out(self.content_upsample_blocks(self.bottle_up(rearrange(z_content, 'B F H W C -> (B F) C H W'))))
        _,_,h,w = z_content_up.shape
        if h > z.size(3):
            border = (h - z.size(3)) // 2
            z_content_up = z_content_up[:, :, border:border+z.size(3), border:border+z.size(4)]
        z_content = rearrange(z_content_up, '(B F) C H W -> (B H W) F C', F=z_content.size(1))
        vt = rearrange(self.cont_emb(z_content), '(B H W) F C -> B C F H W', H=z.size(3), W=z.size(4))
        
        vx = rearrange(self.motion_emb(rearrange(z_motion_x, 'B D F W -> B F W D')), 'B F W C -> B C F W') # shape (b, c', f, w')
        vy = rearrange(self.motion_emb(rearrange(z_motion_y, 'B D F H -> B F H D')), 'B F H C -> B C F H') # shape (b, c', f, h')
        if self.downsample_motion:
            vx = self.up_motion(vx)
            vy = self.up_motion(vy)
        vx = repeat(vx, 'b c f w -> b c f h w', h=z.size(3))
        vy = repeat(vy, 'b c f h -> b c f h w', w=z.size(4))
            
        if only_part == 'content':
            c_plus_m = vt
        elif only_part == 'motion':
            c_plus_m = vx + vy
        else:
            c_plus_m = vt + vx + vy    
        x = self.decoder(c_plus_m)
        return x

    def encode(self, x: Any, return_reg_log: bool = False) -> Any:
        z = self.encoder(x) # shape (b, c', f, h', w')
        if self.shuffle_content:
            b, c, f, h, w = z.shape
            z_shuffled = torch.empty_like(z)
            for i in range(b):
                idx = torch.randperm(f)
                z_shuffled[i] = z[i, :, idx, :, :]
            pre_qformer = rearrange(z_shuffled, 'B C F H W -> (B H W) F C')   
        else: 
            pre_qformer = rearrange(z, 'B C F H W -> (B H W) F C')
        z_content = self.temporal_qformer(pre_qformer) # shape (bhw, f_q, d_q)
        z_content_down = self.bottle_down(self.content_downsample_blocks(self.conv_in(rearrange(z_content, '(B H W) F C -> (B F) C H W', H=z.size(3), W=z.size(4)))))
        z_content = rearrange(z_content_down, '(B F) C H W -> B C F H W', F=z_content.size(1))
        z_content, content_reglog = self.regularization(z_content)
        z_content = rearrange(z_content, 'B C F H W -> B F H W C')
        # intuitively, we can view the z_content as a method to retrieve the content frames (including its nums and dims)
        z_motion_x, z_motion_y = self.get_motion_latent(z)
        z_motion_x, z_motion_x_log = self.regularization(z_motion_x)
        z_motion_y, z_motion_y_log = self.regularization(z_motion_y)
        reg_log = {}
        reg_log['kl_loss'] = content_reglog['kl_loss'] + z_motion_x_log['kl_loss'] + z_motion_y_log['kl_loss']
        if return_reg_log:
            return z, z_content, z_motion_x, z_motion_y, reg_log
        return z, z_content, z_motion_x, z_motion_y

    def get_motion_latent(self, z: torch.Tensor) -> torch.Tensor:
        f = z.size(2)
        if self.downsample_motion:
            z = self.downsample_motion_module(rearrange(z, 'B C F H W -> (B F) C H W'))
            z = rearrange(z, '(B F) C H W -> B C F H W', F=f)
        ux = torch.mean(z, dim=-2) # shape (b, c', f, w')
        uy = torch.mean(z, dim=-1) # shape (b, c', f, h')
        zx = self.motion_head(ux) # shape (b, d, f, w')
        zy = self.motion_head(uy) # shape (b, d, f, h')
  
        return zx, zy
    
    def forward(self, x: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (bs, 3, 17, h, w)
        z, z_content, z_motion_x, z_motion_y, reg_log = self.encode(x, return_reg_log=True)
        # z: shape (b, c', f, h', w')
        
        dec = self.decode(z, z_content, z_motion_x, z_motion_y)
        # dec: (bs, 3, 17, h, w)
        return z, dec, reg_log, z_content, z_motion_x, z_motion_y


    
class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.bs, self.bt = block_size

    def forward(self, x):
        B, C, N, H, W = x.size()
        x = x.view(B, self.bt, self.bs, self.bs, C // ((self.bs ** 2) * self.bt), N,  H, W)  # (B, bs, bs, bs, C//bs^2, N, H, W)
        x = x.permute(0, 4, 5, 1, 6, 2, 7, 3).contiguous()  # (B, C//bs^3, N, bs, H, bs, W, bs)
        x = x.view(B, C // ((self.bs ** 2) * self.bt), N * self.bt,  H * self.bs, W * self.bs)  # (B, C//bs^3, N * bs, H * bs, W * bs)
        # remove the first frame
        if self.bt > 1:
            x = x[:, :, 1:, :, :]
        else:
            x = x
        return x
    
    
from torch.optim.lr_scheduler import _LRScheduler   


class LinearWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, target_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.target_lr = target_lr
        self.total_steps = total_steps
        super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warm-up
            return [base_lr * (self.last_epoch / self.warmup_steps) for base_lr in self.base_lrs]
        elif self.last_epoch < self.total_steps:
            # Constant learning rate
            return [base_lr * (1 - self.last_epoch / self.total_steps) for base_lr in self.base_lrs]
        else:
            return self.base_lrs

class ConstantWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        # self.base_lrs = lr_max
        super(ConstantWarmupScheduler, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warm-up
            return [base_lr * (self.last_epoch / self.warmup_steps) for base_lr in self.base_lrs]
        elif self.last_epoch < self.total_steps:
            # Constant learning rate
            return self.base_lrs         

class LambdaWarmUpCosineScheduler(_LRScheduler):
    """
    note: use with a base_lr of 1.0
    """
    def __init__(self, optimizer, lr_min, lr_max, lr_start, total_steps, warmup_rate = -1, verbosity_interval=0, last_epoch=-1, warmup_steps=-1):
        self.verbosity_interval = verbosity_interval 
        if warmup_rate >= 0: 
            self.lr_warm_up_steps = total_steps * warmup_rate
        elif warmup_steps >= 0:
            self.lr_warm_up_steps = warmup_steps
        else:
            self.lr_warm_up_steps = 0
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_max_decay_steps = total_steps
        super(LambdaWarmUpCosineScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.verbosity_interval > 0:
            if self.last_epoch % self.verbosity_interval == 0: print(f"current step: {self.last_epoch}, recent lr-multiplier: {self.last_lr}")
        if self.last_epoch < self.lr_warm_up_steps:
            lr = (self.lr_max - self.lr_start) / self.lr_warm_up_steps * self.last_epoch + self.lr_start
            self.last_lr = lr 
            return [lr]
        else:
            t = (self.last_epoch - self.lr_warm_up_steps) / (self.lr_max_decay_steps - self.lr_warm_up_steps)
            t = min(t, 1.0)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                    1 + np.cos(t * np.pi))  # a + 0.5 * (b - a) * (1 + cos(pi * t)), where t \in [0, 1], so the lr will be in [a, b]
            self.last_lr = lr
            return [lr]



def instantiate_lrscheduler_from_config(optimizer, config, name='main-LR'):
    """
    Instantiate a learning rate scheduler from a config dict.
    If use timm, must add the following codes to the LightningModule:

    def lr_scheduler_step(self, scheduler, metric):
        if 'timm.scheduler' in self.lr_scheduler_config.target:
            scheduler.step(epoch=self.current_epoch)
        else:
            if metric is None:
                scheduler.step()
            else:
                scheduler.step(metric)
    """
    assert 'target' in config, 'Expected key `target` to instantiate.'
    if ('torch.optim' in config.target) or ('timm.scheduler' in config.target):
        scheduler = get_obj_from_str(config["target"])(optimizer, **config.get("params", dict()))
        lr_scheduler = {
            'scheduler': scheduler,
            'name': name
            }
    else:
        scheduler_init = instantiate_from_config(config)
        scheduler = LambdaLR(optimizer, lr_lambda=scheduler_init.schedule)
        lr_scheduler = {
            'scheduler': LambdaLR(optimizer, lr_lambda=scheduler_init.schedule),
            'name': name,
            'interval': 'step',
            'frequency': 1
            }
    return scheduler



