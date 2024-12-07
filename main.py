import argparse
import datetime
import pytz
import glob
import inspect
import os
import re
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from rich import print
from inspect import Parameter
from typing import Union
from matplotlib import pyplot as plt
from natsort import natsorted
from omegaconf import OmegaConf
from packaging import version
from PIL import Image
from pathlib import Path

import torch
import torch.distributed as dist
import torchvision
import wandb

import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from vidtok.modules.util import exists, instantiate_from_config, isheatmap, print0, seed_anything

MULTINODE_HACKS = True


def default_trainer_args():
    argspec = dict(inspect.signature(Trainer.__init__).parameters)
    argspec.pop("self")
    default_args = {
        param: argspec[param].default
        for param in argspec
        if argspec[param] != Parameter.empty
    }
    return default_args


def get_step_value(folder_name):
    match = re.search(r"step=(\d+)", folder_name)
    if match:
        return int(match.group(1))
    return 0


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "--no_date",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="if True, skip date generation for logdir and only use naming via opt.base or opt.name (+ opt.postfix, optionally)",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=True,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=True,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p", "--project", help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "--seed_rank",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="reset seed every rank on fit start",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    parser.add_argument(
        "--legacy_naming",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="name run based on config file name if true, else by whole path",
    )
    parser.add_argument(
        "--enable_tf32",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="enables the TensorFloat32 format both for matmuls and cuDNN for pytorch 1.12",
    )
    parser.add_argument(
        "--startup",
        type=str,
        default=None,
        help="Startuptime from distributed script",
    )
    parser.add_argument(
        "--wandb",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="log to wandb",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="",
        help="Wandb entity name string",
    )
    parser.add_argument(
        "--wandb_key",
        type=str,
        default="",
        help="Wandb key",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="vidtok",
    )
    parser.add_argument(
        "--wandb_id",
        type=str,
        default=None,
        help="automatically resume from the same wandb id"
        "must be used in combination with --wandb_auto_resume False",
    )
    parser.add_argument(
        "--wandb_auto_resume",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="will find the latest run id in the logdir"
        "if checkpoint_auto_resume is False, wandb_auto_resume will be ignored",
    )
    parser.add_argument(
        "--checkpoint_auto_resume",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="will find the latest checkpoint in the logdir"
        "if checkpoint_auto_resume is False, wandb_auto_resume will be ignored",
    )
    parser.add_argument(
        "--no_base_name",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,  # TODO: later default to True
        help="log to wandb",
    )
    if version.parse(torch.__version__) >= version.parse("2.0.0"):
        parser.add_argument(
            "--resume_from_checkpoint",
            type=str,
            default=None,
            help="single checkpoint file to resume from",
        )
    default_args = default_trainer_args()
    for key in default_args:
        # parameters in the pl.Trainer are passed as --key value
        parser.add_argument("--" + key, default=default_args[key])
    return parser


def get_checkpoint_name(logdir):
    ckpt = os.path.join(logdir, "checkpoints", "last**.ckpt")
    ckpt = natsorted(glob.glob(ckpt))
    print0('available "last" checkpoints:')
    print0(ckpt)
    if len(ckpt) > 1:
        print0("got most recent checkpoint")
        ckpt = sorted(ckpt, key=lambda x: os.path.getmtime(x))[-1]
        print0(f"Most recent ckpt is {ckpt}")
        with open(os.path.join(logdir, "most_recent_ckpt.txt"), "w") as f:
            f.write(ckpt + "\n")
        try:
            version = int(ckpt.split("/")[-1].split("-v")[-1].split(".")[0])
        except Exception as e:
            print0("version confusion but not bad")
            print0(e)
            version = 1
        # version = last_version + 1
    else:
        # in this case, we only have one "last.ckpt"
        ckpt = ckpt[0]
        version = 1
    melk_ckpt_name = f"last-v{version}.ckpt"
    print0(f"Current melk ckpt name: {melk_ckpt_name}")
    return ckpt, melk_ckpt_name


class SetupCallback(Callback):
    def __init__(
        self,
        resume,
        now,
        logdir,
        ckptdir,
        cfgdir,
        config,
        lightning_config,
        debug,
        save_ckpt_on_exception=False,
        ckpt_name=None,
        seed=None,
        seed_rank=False,
    ):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config
        self.debug = debug
        self.save_ckpt_on_exception = save_ckpt_on_exception
        self.ckpt_name = ckpt_name
        self.seed = seed
        self.seed_rank = seed_rank

    def on_exception(self, trainer: pl.Trainer, pl_module, exception):
        if self.save_ckpt_on_exception and (not self.debug) and (trainer.global_rank == 0):
            print0(f"[bold red]\[main][SetupCallback][/bold red] Saving checkpoint to {self.ckptdir}")
            if self.ckpt_name is None:
                ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            else:
                ckpt_path = os.path.join(self.ckptdir, self.ckpt_name)
            trainer.save_checkpoint(ckpt_path)

    def on_fit_start(self, trainer, pl_module):
        if self.seed_rank:
            # current_seed = torch.initial_seed()
            seed_anything(self.seed + trainer.global_rank)
            print(f"[bold red]\[main][SetupCallback][/bold red] Rank {trainer.global_rank}: Reset GLOBAL seed to {self.seed + trainer.global_rank}")
        elif hasattr(pl_module, "set_seed") and callable(pl_module.set_seed):
            pl_module.set_seed(self.seed)
            print0(f"[bold red]\[main][SetupCallback][/bold red] Set pl_module seed to {self.seed} with pl_module.set_seed")
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            print0(f"[bold red]\[main][SetupCallback][/bold red] Creating logdir: {self.logdir}, ckptdir: {self.ckptdir}, cfgdir: {self.cfgdir}")
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if (
                    "metrics_over_trainsteps_checkpoint"
                    in self.lightning_config["callbacks"]
                ):
                    os.makedirs(
                        os.path.join(self.ckptdir, "trainstep_checkpoints"),
                        exist_ok=True,
                    )
            print0("[bold red]\[main][SetupCallback][/bold red] Project config")
            print0(OmegaConf.to_yaml(self.config))
            if MULTINODE_HACKS and not self.debug:
                import time
                time.sleep(5)
            OmegaConf.save(
                self.config,
                os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)),
            )

            print0("[bold red]\[main][SetupCallback][/bold red] Lightning config")
            print0(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(
                OmegaConf.create({"lightning": self.lightning_config}),
                os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)),
            )

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not MULTINODE_HACKS and not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    def __init__(
        self,
        batch_frequency,
        max_samples,
        clamp=True,
        increase_log_steps=True,
        rescale=True,
        disabled=True,
        log_on_batch_idx=False,
        log_first_step=False,
        log_images_kwargs=None,
        log_before_first_step=False,
        enable_autocast=True,
    ):
        super().__init__()
        self.enable_autocast = enable_autocast
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_samples = max_samples
        self.log_steps = [2**n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.log_before_first_step = log_before_first_step

    @rank_zero_only
    def log_local(
        self,
        save_dir,
        split,
        images,
        global_step,
        current_epoch,
        batch_idx,
        pl_module: Union[None, pl.LightningModule] = None,
    ):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            if isheatmap(images[k]):
                fig, ax = plt.subplots()
                ax = ax.matshow(
                    images[k].cpu().numpy(), cmap="hot", interpolation="lanczos"
                )
                plt.colorbar(ax)
                plt.axis("off")

                filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                    k, global_step, current_epoch, batch_idx
                )
                os.makedirs(root, exist_ok=True)
                path = os.path.join(root, filename)
                plt.savefig(path)
                plt.close()
                # TODO: support wandb
            else:
                grid = torchvision.utils.make_grid(images[k], nrow=4)
                if self.rescale:
                    grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                grid = grid.numpy()
                grid = (grid * 255).astype(np.uint8)
                filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                    k, global_step, current_epoch, batch_idx
                )
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                img = Image.fromarray(grid)
                img.save(path)
                if exists(pl_module):
                    assert isinstance(
                        pl_module.logger, WandbLogger
                    ), "logger_log_image only supports WandbLogger currently"
                    pl_module.logger.log_image(
                        key=f"{split}/{k}",
                        images=[
                            img,
                        ],
                        step=pl_module.global_step,
                    )

    @rank_zero_only
    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (
            self.check_frequency(check_idx)
            and hasattr(pl_module, "log_images")  # batch_idx % self.batch_freq == 0
            and callable(pl_module.log_images)
            and self.max_samples > 0
        ):
            logger = type(pl_module.logger)
            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            gpu_autocast_kwargs = {
                "enabled": self.enable_autocast,  # torch.is_autocast_enabled(),
                "dtype": torch.get_autocast_gpu_dtype(),
                "cache_enabled": torch.is_autocast_cache_enabled(),
            }
            with torch.no_grad(), torch.cuda.amp.autocast(**gpu_autocast_kwargs):
                images = pl_module.log_images(
                    batch, split=split, **self.log_images_kwargs
                )

            for k in images:
                N = min(images[k].shape[0], self.max_samples)
                if not isheatmap(images[k]):
                    images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().float().cpu()
                    if self.clamp and not isheatmap(images[k]):
                        images[k] = torch.clamp(images[k], -1.0, 1.0)

            self.log_local(
                pl_module.logger.save_dir,
                split,
                images,
                pl_module.global_step,
                pl_module.current_epoch,
                batch_idx,
                pl_module=pl_module
                if isinstance(pl_module.logger, WandbLogger)
                else None,
            )

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
            check_idx > 0 or self.log_first_step
        ):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print0("[bold red]\[main][ImageLogger][/bold red]", e)
                pass
            return True
        return False

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    @rank_zero_only
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.log_before_first_step and pl_module.global_step == 0:
            print0(f"[bold red]\[main][ImageLogger][/bold red] {self.__class__.__name__}: logging before training")
            self.log_img(pl_module, batch, batch_idx, split="train")

    @rank_zero_only
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs
    ):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, "calibrate_grad_norm"):
            if (
                pl_module.calibrate_grad_norm and batch_idx % 25 == 0
            ) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


@rank_zero_only
def init_wandb(save_dir, opt, config, group_name, name_str):
    print0(f"[bold red]\[main][init_wandb][/bold red] Creating WANDB_DIR: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    # os.environ["WANDB_DIR"] = save_dir
    gitcmd = f'git config --global --add safe.directory {os.path.dirname(os.path.abspath(__file__))}'
    os.system(gitcmd)
    print0(f"[bold red]\[main][init_wandb][/bold red] wandb_id is set to {opt.wandb_id}")
    wandb_id = opt.wandb_id if opt.wandb_id is not None else name_str

    if not wandb.api.api_key:
        wandb.login(key=opt.wandb_key)
    if opt.debug:
        wandb.init(project=opt.wandb_project, mode="offline", group=group_name)
    else:
        wandb.init(
            project=opt.wandb_project,
            entity=opt.wandb_entity,
            config=dict(config),
            group=group_name,
            name=name_str,
            resume='auto',
            id=wandb_id,
        )


if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()

    opt, unknown = parser.parse_known_args()

    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    melk_ckpt_name = None
    name = None
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
            _, melk_ckpt_name = get_checkpoint_name(logdir)
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt, melk_ckpt_name = get_checkpoint_name(logdir)

        print0("-" * 80)
        print0(f'[bold red][main][/bold red] Resuming from checkpoint "{ckpt}"')

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            if opt.no_base_name:
                name = ""
            else:
                if opt.legacy_naming:
                    cfg_fname = os.path.split(opt.base[0])[-1]
                    cfg_name = os.path.splitext(cfg_fname)[0]
                else:
                    assert "configs" in os.path.split(opt.base[0])[0], os.path.split(
                        opt.base[0]
                    )[0]
                    cfg_path = os.path.split(opt.base[0])[0].split(os.sep)[
                        os.path.split(opt.base[0])[0].split(os.sep).index("configs")
                        + 1 :
                    ]  # cut away the first one (we assert all configs are in "configs")
                    cfg_name = os.path.splitext(os.path.split(opt.base[0])[-1])[0]
                    cfg_name = "-".join(cfg_path) + f"-{cfg_name}"
                name = "_" + cfg_name
        else:
            name = ""
        # automatic resume last checkpoint if available
        if os.path.exists(opt.logdir):
            auto_resumed = False
            for sub_dir in sorted(os.listdir(opt.logdir)):
                if sub_dir.endswith(name + opt.postfix):
                    ## checkpoint resume
                    if opt.checkpoint_auto_resume and not opt.debug:
                        checkpoint_dir = os.path.join(opt.logdir, sub_dir, "checkpoints")
                        # Use the max step checkpoint file
                        ckpt_files1 = glob.glob(os.path.join(checkpoint_dir, "*/*.ckpt"))
                        ckpt_files2 = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
                        ckpt_files = ckpt_files1 + ckpt_files2
                        ckpt_files.sort(key=get_step_value, reverse=True)
                        if ckpt_files:
                            ckpt = ckpt_files[0]
                        else:
                            # If no checkpoint files found, use a random initialized model
                            ckpt = None
                        if ckpt is not None and os.path.isfile(ckpt):
                            opt.resume_from_checkpoint = ckpt
                            auto_resumed = True
                            # print0("-" * 80)
                            print0(f"[bold red]\[main][/bold red] Find previous log dir and checkpoint: {ckpt}")
                            ## wandb resume
                            if opt.wandb_auto_resume:
                                wandb_dir = Path(os.path.join(opt.logdir, sub_dir)) / "wandb"
                                if wandb_dir.exists() and any((wandb_dir / "latest-run").iterdir()):
                                    # Parse unique `run_id` from the `.wandb.` file...
                                    wandb_fns = [f.name for f in (wandb_dir / "latest-run").iterdir() if f.name.endswith(".wandb")]
                                    assert len(wandb_fns) == 1, f"There should only be 1 `.wandb.` file... found {len(wandb_fns)}!"
                                    # Regex Match on `run-{id}.wandb`
                                    opt.wandb_id = re.search("run-(.+?).wandb", wandb_fns[0]).group(1)
                                    # print0("-" * 80)
                                    print0(f"[bold red]\[main][/bold red] Find previous wandb run id: {opt.wandb_id}")
            if auto_resumed:
                print0(f"[bold red]\[main][/bold red] Auto-resuming from checkpoint: {opt.resume_from_checkpoint} and wandb id: {opt.wandb_id}")
                ckpt_basename = os.path.basename(opt.resume_from_checkpoint)
                seed_str = ''.join(re.findall(r'\d+', ckpt_basename))
                if len(seed_str) > 0:
                    opt.seed = int(seed_str)
                    print0(f"[bold red]\[main][/bold red] Auto-reseting seed to {opt.seed} from checkpoint name")

        if not opt.no_date:
            nowname = now + name + opt.postfix
        else:
            nowname = name + opt.postfix
            if nowname.startswith("_"):
                nowname = nowname[1:]
        logdir = os.path.join(opt.logdir, nowname)
        print0(f"[bold red]\[main][/bold red] LOGDIR: {logdir}")

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    if not opt.seed_rank:
        seed_everything(opt.seed, workers=True)  # torch.initial_seed()

    # move before model init, in case a torch.compile(...) is called somewhere
    if opt.enable_tf32:
        # pt_version = version.parse(torch.__version__)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print0(f"[bold red]\[main][/bold red] Enabling TF32 for PyTorch {torch.__version__}")
    else:
        print0(f"[bold red]\[main][/bold red] Using default TF32 settings for PyTorch {torch.__version__}:")
        print0(f"[bold red]\[main][/bold red] torch.backends.cuda.matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32}")
        print0(f"[bold red]\[main][/bold red] torch.backends.cudnn.allow_tf32={torch.backends.cudnn.allow_tf32}")

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        # deal with the unknown args, e.g., --model.base_learning_rate=1.0e-4
        for i, u in enumerate(unknown):
            if u.startswith("--"):
                unknown[i] = u[2:]
        # merge all configs and cli args
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        print0("-" * 80)
        print0(f"[bold red]\[main][/bold red] Merged input config: {config}")
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())

        # debug: default to one node
        if opt.debug:
            trainer_config["num_nodes"] = 1

        # default profiler
        trainer_config["profiler"] = None if not opt.debug else "simple"

        # default to gpu
        trainer_config["accelerator"] = "gpu"
        #
        standard_args = default_trainer_args()
        for k in standard_args:
            if getattr(opt, k) != standard_args[k]:
                trainer_config[k] = getattr(opt, k)

        if not "devices" in trainer_config and trainer_config["accelerator"] != "gpu":
            del trainer_config["accelerator"]
            cpu = True
        else:
            gpuinfo = trainer_config["devices"]
            print0(f"[bold red]\[main][/bold red] Running on {gpuinfo} GPUs")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        model = instantiate_from_config(config.model)

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        default_logger_cfgs = {
            "wandb": {
                "target": "lightning.pytorch.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                    "project": opt.wandb_project,
                    "log_model": False,
                    "entity": opt.wandb_entity,
                },
            },
            "csv": {
                "target": "lightning.pytorch.loggers.CSVLogger",
                "params": {
                    "name": "testtube",  # hack for sbord fanatics
                    "save_dir": logdir,
                },
            },
            "tensorboard": {
                "target": "lightning.pytorch.loggers.TensorBoardLogger",
                "params": {
                    "save_dir": logdir,
                    "name": 'tensorboard',
                    "version": nowname,
                }
            },
        }
        default_logger_cfg = default_logger_cfgs["wandb" if opt.wandb else "tensorboard"]
        if opt.wandb:
            # change once leaving "swiffer" config directory
            try:
                group_name = nowname.split(now)[-1].split("-")[1]
            except:
                group_name = nowname
            default_logger_cfg["params"]["group"] = group_name

            wandb_save_dir = os.path.join(os.getcwd(), logdir)
            os.environ["WANDB_DIR"] = wandb_save_dir

            init_wandb(
                wandb_save_dir,
                opt=opt,
                group_name=group_name,
                config=config,
                name_str=nowname,
            )
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        ckpt_resume_path = opt.resume_from_checkpoint

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "lightning.pytorch.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:04}-{step:08}",  # "epoch={epoch:06}-step={step:07}"
                "verbose": True,
                "save_last": True,
                "auto_insert_metric_name": True,
            },
        }
        if hasattr(model, "monitor"):
            print0(f"[bold red]\[main][/bold red] Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = 3

        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg = OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        print0("-" * 80)
        print0(f"[bold red]\[main][/bold red] Merged modelckpt-cfg: {modelckpt_cfg}")

        # https://pytorch-lightning.readthedocs.io/en/stable/extensions/strategy.html
        # default to ddp if not further specified
        default_strategy_config = {"target": "lightning.pytorch.strategies.DDPStrategy"}

        if "strategy" in lightning_config:
            strategy_cfg = lightning_config.strategy
        else:
            strategy_cfg = OmegaConf.create()
            default_strategy_config["params"] = {
                "find_unused_parameters": False,
                # "static_graph": True,
                # "ddp_comm_hook": default.fp16_compress_hook  # experiment with this, also for DDPSharded
            }
        strategy_cfg = OmegaConf.merge(default_strategy_config, strategy_cfg)
        print0("-" * 80)
        print0(f"[bold red]\[main][/bold red] strategy config: {strategy_cfg}")
        trainer_kwargs["strategy"] = instantiate_from_config(strategy_cfg)
        if hasattr(trainer_kwargs["strategy"], "_timeout"):
            trainer_kwargs["strategy"]._timeout = datetime.timedelta(seconds=5400)  # 3600s = 1h

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                    "debug": opt.debug,
                    "ckpt_name": melk_ckpt_name,
                    "seed": opt.seed,
                    "seed_rank": opt.seed_rank
                },
            },
            "image_logger": {
                "target": "main.ImageLogger",
                "params": {"batch_frequency": 1000, "max_samples": 4, "clamp": True},
            },
            "learning_rate_logger": {
                "target": "lightning.pytorch.callbacks.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    # "log_momentum": True
                },
            },
        }
        if version.parse(pl.__version__) >= version.parse("1.4.0"):
            default_callbacks_cfg.update({"checkpoint_callback": modelckpt_cfg})

        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()

        if "metrics_over_trainsteps_checkpoint" in callbacks_cfg:
            print0(
                "[bold red]\[main][/bold red] Caution: Saving checkpoints every n train steps without deleting. This might require some free space."
            )
            default_metrics_over_trainsteps_ckpt_dict = {
                "metrics_over_trainsteps_checkpoint": {
                    "target": "lightning.pytorch.callbacks.ModelCheckpoint",
                    "params": {
                        "dirpath": os.path.join(ckptdir, "trainstep_checkpoints"),
                        "filename": "{epoch:04}-{step:08}",  # "{epoch:06}-{step:09}"
                        "verbose": True,
                        "save_top_k": -1,
                        "every_n_train_steps": 10000,
                        "save_weights_only": True,
                    },
                }
            }
            default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        if "ignore_keys_callback" in callbacks_cfg and ckpt_resume_path is not None:
            callbacks_cfg.ignore_keys_callback.params["ckpt_path"] = ckpt_resume_path
        elif "ignore_keys_callback" in callbacks_cfg:
            del callbacks_cfg["ignore_keys_callback"]

        trainer_kwargs["callbacks"] = [
            instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg
        ]
        if not "plugins" in trainer_kwargs:
            trainer_kwargs["plugins"] = list()

        # cmd line trainer args (which are in trainer_opt) have always priority over config-trainer-args (which are in trainer_kwargs)
        trainer_opt = vars(trainer_opt)
        trainer_kwargs = {
            key: val for key, val in trainer_kwargs.items() if key not in trainer_opt
        }
        trainer = Trainer(**trainer_opt, **trainer_kwargs)

        trainer.logdir = logdir

        # data
        if ((not opt.train) or opt.debug) and hasattr(config.data.params, "validation"):
            config.data.params.train = config.data.params.validation
            print0("[bold red]\[main][/bold red] Using validation data as training data for fast loading.")
        data = instantiate_from_config(config.data)
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        data.prepare_data()
        # data.setup()
        try:
            for k in data.datasets:
                print0(
                    f"[bold red]\[main][/bold red] {k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}"
                )
        except:
            print0("[bold red]\[main][/bold red] datasets not yet initialized.")

        # configure learning rate
        if "batch_size" in config.data.params:
            bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        else:
            bs, base_lr = (
                config.data.params.train.loader.batch_size,
                config.model.base_learning_rate,
            )
        if not cpu:
            # add for different device input type
            if isinstance(lightning_config.trainer.devices, int):
                ngpu = lightning_config.trainer.devices
            elif isinstance(lightning_config.trainer.devices, list):
                ngpu = len(lightning_config.trainer.devices)
            elif isinstance(lightning_config.trainer.devices, str):
                ngpu = len(lightning_config.trainer.devices.strip(",").split(","))
        else:
            ngpu = 1
        if "accumulate_grad_batches" in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        print0(f"[bold red]\[main][/bold red] accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches

        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            print0(
                "[bold red]\[main][/bold red] Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr
                )
            )
        else:
            model.learning_rate = base_lr
            print0("[bold red]\[main][/bold red] NOT using learning rate scaling")
            print0(f"[bold red]\[main][/bold red] Setting learning rate to {model.learning_rate:.2e}")

        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                melkdir = os.path.join(logdir, "melk")
                os.makedirs(melkdir, exist_ok=True)
                print0(f"[bold red]\[main][/bold red] Saving checkpoint to {melkdir}")
                if melk_ckpt_name is None:
                    ckpt_path = os.path.join(melkdir, "last.ckpt")
                else:
                    ckpt_path = os.path.join(melkdir, melk_ckpt_name)
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb
                pudb.set_trace()

        import signal
        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            try:
                trainer.fit(model, data, ckpt_path=ckpt_resume_path)
                print0(f"[bold red]\[main][/bold red] Finish training with logdir: {logdir}")
            except Exception as e:
                print(f"")
                print(f"[bold red]\[main][/bold red] Exception: {e}")
                print(f"[bold red]\[main][/bold red] Beijing Time {datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai'))}")
                if not opt.debug:
                    melk()
                raise
        else:
            trainer.validate(model, data, ckpt_path=ckpt_resume_path)
            exit()
        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)
    except RuntimeError as err:
        if MULTINODE_HACKS:
            import datetime
            import os
            import socket
            import requests

            device = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
            hostname = socket.gethostname()
            ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            resp = requests.get("http://169.254.169.254/latest/meta-data/instance-id")
            print(
                f"[bold red]\[main][/bold red] ERROR at {ts} on {hostname}/{resp.text} (CUDA_VISIBLE_DEVICES={device}): {type(err).__name__}: {err}",
                flush=True,
            )
        raise err
    except Exception:
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            # debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)

        if opt.wandb:
            wandb.finish()

        # clean up
        # dist.barrier()
        # torch.cuda.empty_cache()
        dist.destroy_process_group()

        if trainer.global_rank == 0 and opt.debug:
            print0(f"[bold red]\[main][/bold red] Current logdir: {logdir}")
            # print0(f"[bold red]\[main][/bold red] Profiler summary:")
            # print(trainer.profiler.summary())
            print0(f"[bold red]\[main][/bold red] Memory summary:")
            num_params = sum([p.numel() for p in model.parameters()])
            print0(f"[bold red]\[main][/bold red] Expected bf16 memory usage from params: {num_params * 2 / 1e9:.2f} GB")
            print0(f"[bold red]\[main][/bold red] Current memory usage with model on device {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
            # trainer.print(torch.cuda.memory_summary())
