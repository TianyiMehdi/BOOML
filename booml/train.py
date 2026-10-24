import os
import sys

if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    os.environ["HYDRA_FULL_ERROR"]="1"
os.environ["LAZY_LEGACY_OP"] = "0"
import warnings

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import hydra
from termcolor import colored

from booml.common.parser import parse_cfg
from booml.common.seed import set_seed
from booml.common.buffer import Buffer
from booml.envs import make_env
from booml.booml_alg import BOOML
from booml.boom_alg import BOOM
from booml.booml_alg_adaptive import BOOML_adaptive
from booml.trainer.online_trainer import OnlineTrainer
from booml.common.logger import Logger

torch.backends.cudnn.benchmark = True

@hydra.main(config_name="config", config_path=".")
def train(cfg: dict):
    assert cfg.steps > 0, "Must train for at least 1 step."
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    print(colored("Work dir:", "yellow", attrs=["bold"]), cfg.work_dir)

    trainer_cls = OnlineTrainer
    
    if cfg.exp_name == 'booml':
        trainer = trainer_cls(
            cfg=cfg,
            env=make_env(cfg),
            agent=BOOML(cfg),
            buffer=Buffer(cfg),
            logger=Logger(cfg),
        )
    elif cfg.exp_name == 'adaptive':
        trainer = trainer_cls(
            cfg=cfg,
            env=make_env(cfg),
            agent=BOOML_adaptive(cfg),
            buffer=Buffer(cfg),
            logger=Logger(cfg),
        )
    elif cfg.exp_name == 'boom':
        trainer = trainer_cls(
            cfg=cfg,
            env=make_env(cfg),
            agent=BOOM(cfg),
            buffer=Buffer(cfg),
            logger=Logger(cfg),
        )
    else:
        raise ValueError(f"Unknown exp_name: {cfg.exp_name}")
    trainer.train()
    print("\nTraining completed successfully")


if __name__ == "__main__":
    train()
