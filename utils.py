from random import uniform as randfloat

import gym
import numpy as np
from ray.rllib import MultiAgentEnv
import soccer_twos


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    A RLLib wrapper so our env can inherit from MultiAgentEnv.
    """

    # pass

    def __init__(self, env):
        super().__init__(env)
        self.prev_ball_vel = None

    def reset(self):
        self.prev_ball_vel = None
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        shaped = self._shaping(info)
        combined = {aid: reward[aid] + shaped[aid] for aid in reward}
        return obs, combined, done, info

    def _shaping(self, info) -> dict:
        TEAM_0_GOAL = -13.0
        TEAM_1_GOAL = 13.0

        shaped = {agent_id: 0.0 for agent_id in info}

        def _read_ball_snapshot(state_map):
            for _, snapshot in state_map.items():
                ball_blob = snapshot.get("ball_info")
                if ball_blob:
                    return np.array(ball_blob["position"]), np.array(ball_blob["velocity"])
            return None, None

        ball_pos, ball_vel = _read_ball_snapshot(info)
        if ball_pos is None:
            return shaped

        for squad_base in (2, 0):
            squad_ids = [squad_base, squad_base + 1]
            if not all("player_info" in info.get(i, {}) for i in squad_ids):
                continue
            positions_by_id = {i: np.array(info[i]["player_info"]["position"]) for i in squad_ids}
            distances_to_ball = [np.linalg.norm(positions_by_id[i] - ball_pos) for i in squad_ids]
            passer_index = int(np.argmin(distances_to_ball))
            passer_id = squad_ids[passer_index]
            receiver_id = squad_ids[1] if passer_id == squad_ids[0] else squad_ids[0]

            receiver_pos = positions_by_id[receiver_id]
            toward_receiver = receiver_pos - ball_pos
            receiver_range = np.linalg.norm(toward_receiver) + 1e-6
            pass_score = np.dot(ball_vel, toward_receiver / receiver_range)

            if pass_score > 0.5 and distances_to_ball[passer_index] < 1.5:
                shaped[passer_id] += 0.001 * pass_score

        for squad_base in (2, 0):
            left_id, right_id = squad_base, squad_base + 1
            left_info = info.get(left_id, {})
            right_info = info.get(right_id, {})
            if "player_info" in left_info and "player_info" in right_info:
                left_pos = np.array(left_info["player_info"]["position"])
                right_pos = np.array(right_info["player_info"]["position"])
                spread = min(float(np.linalg.norm(left_pos - right_pos)), 5.0)
                shaped[left_id] += 0.0001 * spread
                shaped[right_id] += 0.0001 * spread

        for aid, ainfo in info.items():
            player_state = ainfo.get("player_info")
            if not player_state:
                continue

            player_pos = np.array(player_state["position"])
            player_vel = np.array(player_state["velocity"])
            defended_goal_x = TEAM_0_GOAL if aid < 2 else TEAM_1_GOAL
            drive_dir = 1.0 if aid < 2 else -1.0

            vector_to_ball = ball_pos - player_pos
            ball_distance = np.linalg.norm(vector_to_ball) + 1e-6

            approach_score = np.dot(player_vel, vector_to_ball / ball_distance)
            shaped[aid] += 0.001 * approach_score

            if self.prev_ball_vel is not None:
                speed_delta = np.linalg.norm(ball_vel) - np.linalg.norm(self.prev_ball_vel)
                if speed_delta > 0 and ball_distance < 1.5:
                    shaped[aid] += 0.001 * speed_delta

            shaped[aid] += 0.00015 * ball_vel[0] * drive_dir

            hazard = max(0.0, 1.0 - abs(ball_pos[0] - defended_goal_x) / 5.0)
            shaped[aid] -= 0.002 * hazard

            shaped[aid] -= 0.00003

        self.prev_ball_vel = ball_vel

        return shaped



def create_rllib_env(env_config: dict = {}):
    """
    Creates a RLLib environment and prepares it to be instantiated by Ray workers.
    Args:
        env_config: configuration for the environment.
            You may specify the following keys:
            - variation: one of soccer_twos.EnvType. Defaults to EnvType.multiagent_player.
            - opponent_policy: a Callable for your agent to train against. Defaults to a random policy.
    """
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    env = soccer_twos.make(**env_config)
    # env = TransitionRecorderWrapper(env)
    if "multiagent" in env_config and not env_config["multiagent"]:
        # is multiagent by default, is only disabled if explicitly set to False
        return env
    return RLLibWrapper(env)


def sample_vec(range_dict):
    return [
        randfloat(range_dict["x"][0], range_dict["x"][1]),
        randfloat(range_dict["y"][0], range_dict["y"][1]),
    ]


def sample_val(range_tpl):
    return randfloat(range_tpl[0], range_tpl[1])


def sample_pos_vel(range_dict):
    _s = {}
    if "position" in range_dict:
        _s["position"] = sample_vec(range_dict["position"])
    if "velocity" in range_dict:
        _s["velocity"] = sample_vec(range_dict["velocity"])
    return _s


def sample_player(range_dict):
    _s = sample_pos_vel(range_dict)
    if "rotation_y" in range_dict:
        _s["rotation_y"] = sample_val(range_dict["rotation_y"])
    return _s
