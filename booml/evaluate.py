import os
import sys
from time import perf_counter

if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["LAZY_LEGACY_OP"] = "0"

import warnings

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import hydra
import numpy as np
import torch
from termcolor import colored

from booml.common.parser import parse_cfg
from booml.common.seed import set_seed
from booml.envs import make_env
from booml.boom_alg import BOOM
from booml.booml_alg import BOOML
from booml.booml_alg_adaptive import BOOML_adaptive

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def _make_agent(cfg):
    if cfg.exp_name == "booml":
        return BOOML(cfg)
    if cfg.exp_name == "adaptive":
        return BOOML_adaptive(cfg)
    if cfg.exp_name == "boom":
        return BOOM(cfg)
    raise ValueError(f"Unknown exp_name: {cfg.exp_name}")


@hydra.main(config_name="config", config_path=".")
def evaluate(cfg: dict):
    assert cfg.eval_episodes > 0, "Must evaluate at least 1 episode."
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)

    print("pid:", os.getpid(), flush=True)
    print(colored(f"Task: {cfg.task}", "blue", attrs=["bold"]))
    print(colored(f"Agent: {cfg.exp_name}", "blue", attrs=["bold"]))

    env = make_env(cfg)
    agent = _make_agent(cfg)

    checkpoint = None
    try:
        checkpoint = cfg.checkpoint
    except Exception:
        checkpoint = None

    if checkpoint not in (None, "???"):
        assert os.path.exists(checkpoint), f"Checkpoint {checkpoint} not found!"
        agent.load(checkpoint)
        print(colored(f"Checkpoint: {checkpoint}", "blue", attrs=["bold"]))
    else:
        print(colored("Checkpoint: not provided, using current model weights.", "red", attrs=["bold"]))

    ep_rewards, ep_successes = [], []
    act_time_total = 0.0
    act_calls = 0

    print(colored(f"Evaluating for {cfg.eval_episodes} episodes...", "yellow", attrs=["bold"]))
    for i in range(cfg.eval_episodes):
        obs, done, ep_reward, t = env.reset()[0], False, 0.0, 0
        while not done:
            t_act_start = perf_counter()
            action, _, _ = agent.act(obs, t0=(t == 0), eval_mode=True)
            act_time_total += perf_counter() - t_act_start
            act_calls += 1

            obs, reward, done, truncated, info = env.step(action.detach())
            done = done or truncated
            ep_reward += reward
            t += 1

        ep_rewards.append(ep_reward)
        ep_successes.append(info["success"])

    ep_rewards_std = np.std(ep_rewards)
    ep_rewards_mean = np.mean(ep_rewards)
    ep_successes_mean = np.mean(ep_successes)

    print(
        colored(
            f"  {cfg.task:<22}\tR: {ep_rewards_mean:.01f} ± {ep_rewards_std:.01f}\tS: {ep_successes_mean:.02f}",
            "yellow",
        )
    )
    if act_calls > 0:
        print(
            colored(
                f"Average agent.act time: {(act_time_total / act_calls) * 1000:.3f} ms ({act_calls} calls)",
                "cyan",
                attrs=["bold"],
            )
        )


if __name__ == "__main__":
    evaluate()
