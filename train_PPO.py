import os
import pickle
import numpy as np

import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from utils import create_rllib_env


NUM_ENVS_PER_WORKER = 3

# ── Paths ─────────────────────────────────────────────────────────────────────
CEIA_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ceia_baseline_agent/ray_results/PPO_selfplay_twos/"
    "PPO_Soccer_f475e_00000_0_2021-09-19_15-54-02/checkpoint_002449/checkpoint-2449",
)

RESTORE_CHECKPOINT = (
    "./ray_results/PPO_large/PPO_Soccer_40d48_00000_0_2026-04-23_09-56-41/checkpoint_000203/checkpoint-203"
)
# ─────────────────────────────────────────────────────────────────────────────


def policy_mapping_fn(agent_id, *args, **kwargs):
    episode = None
    if args and hasattr(args[0], 'user_data'):
        episode = args[0]
    train_team = episode.user_data.get("train_team", 0) if episode else 0
    return "default" if agent_id // 2 == train_team else "opponent"


def _load_weights(checkpoint_path: str, policy_name: str = "default") -> dict:
    """Extract policy weights from a Ray checkpoint file."""
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    worker_state = pickle.loads(data["worker"])
    state = worker_state["state"]
    if policy_name not in state:
        policy_name = list(state.keys())[0]
    return {k: v for k, v in state[policy_name].items() if k != "_optimizer_variables"}


class LoadWeights(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self._initialized = False

    def on_episode_start(self, *, episode, **kwargs):
        episode.user_data["train_team"] = np.random.randint(2)

    def on_train_result(self, **info):
        if self._initialized:
            return
        trainer = info["trainer"]
        try:
            weights = _load_weights(CEIA_CHECKPOINT)
            agent_weights = _load_weights(RESTORE_CHECKPOINT)
            trainer.set_weights({"default": agent_weights, "opponent": weights})
        except Exception as e:
            print(f"Error loading weights: {e}")
            raise e
        self._initialized = True


if __name__ == "__main__":
    ray.init()

    tune.registry.register_env("Soccer", create_rllib_env)
    tmp = create_rllib_env()
    obs_space = tmp.observation_space
    act_space = tmp.action_space
    tmp.close()

    fragment_len = 2000
    n_workers = 8
    total_rollout_steps = n_workers * NUM_ENVS_PER_WORKER * fragment_len

    analysis = tune.run(
        "PPO",
        name="PPO",
        config={
            "num_gpus": 0,
            "num_workers": n_workers,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": LoadWeights,
            "multiagent": {
                "policies": {
                    "default": (None, obs_space, act_space, {}),
                    "opponent": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": policy_mapping_fn,
                "policies_to_train": ["default"],
            },
            "env": "Soccer",
            "env_config": {"num_envs_per_worker": NUM_ENVS_PER_WORKER,
                           "base_port": 4271},
            "model": {
                "vf_share_layers": False,
                "fcnet_hiddens": [512, 256],
                "fcnet_activation": "relu",
            },
            "clip_param": 0.2,
            "lambda": 0.95,
            "lr_schedule": [
                [0, 3e-4],
                [20_000_000, 1e-4],
                [50_000_000, 3e-5],
            ],
            "vf_loss_coeff": 2.0,
            "entropy_coeff_schedule": [
                [0, 0.005],
                [20000000, 0.001],
                [50000000, 0.0],
            ],
            "grad_clip": 0.5,
            "train_batch_size": total_rollout_steps,
            "sgd_minibatch_size": 512,
            "num_sgd_iter": 10,
            "rollout_fragment_length": fragment_len,
            "batch_mode": "complete_episodes",
        },
        stop={
            "timesteps_total": 100000000,
            "time_total_s": 70200,
        },
        checkpoint_freq=10,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        restore=RESTORE_CHECKPOINT,
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    best_ckpt = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_trial)
    print(best_ckpt)
    print("Done training")
