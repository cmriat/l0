# Copyright (c) 2022–2025 China Merchants Research Institute of Advanced Technology Corporation and its Affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Advantage calculation module for reinforcement learning agents."""

from enum import Enum

import torch
from verl import DataProto
from config_dataclass import ConfigDataclass
from verl.trainer.ppo.core_algos import compute_gae_advantage_return
from verl.trainer.ppo.ray_trainer import compute_response_mask

from l0.utils import collect_trajectory_data

from . import core_algos as bi_level_core_algos


def extract_token_level_rewards(step):
    return step.batch["token_level_rewards"]


def extract_step_level_rewards(step):
    return step.batch["step_level_rewards"]


def extract_token_level_values(step):
    if "token_level_values" not in step.batch:
        return None
    return step.batch["token_level_values"]


def extract_step_level_values(step):
    if "step_level_values" not in step.batch:
        return None
    return step.batch["step_level_values"]


def extract_response_mask(step):
    return step.batch["response_mask"]


FIELDS_TO_COLLECT = {
    "token_level_rewards": extract_token_level_rewards,
    "step_level_rewards": extract_step_level_rewards,
    "token_level_values": extract_token_level_values,
    "step_level_values": extract_step_level_values,
    "response_mask": extract_response_mask,
}


def step_level_mask_whiten(step_level_advantages: list[list[float]], sigma=1e-8) -> list[list[float]]:
    """Whiten step-level advantages."""
    if len(step_level_advantages) == 0:
        return step_level_advantages

    flatten_advantages = flatten_advantages = [adv for traj in step_level_advantages for adv in traj]

    # Compute mean and std
    mean = torch.mean(torch.tensor(flatten_advantages, dtype=torch.float32)).item()
    std = torch.std(torch.tensor(flatten_advantages, dtype=torch.float32)).item() + sigma

    # Whiten the advantages
    whitened_step_level_advantages = [[(adv - mean) / std for adv in traj] for traj in step_level_advantages]

    return whitened_step_level_advantages


class StepLevelAdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator."""

    GAE = "gae"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"


class TokenLevelAdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator."""

    GAE = "gae"
    TRIVIAL = "trivial"


def compute_step_level_advantage(trajectories, config: ConfigDataclass) -> tuple[list[list[float]], list[list[float]]]:
    """Compute step-level advantage and returns."""
    if config.step_level_adv_estimator == StepLevelAdvantageEstimator.GAE:
        return bi_level_core_algos.step_level_gae_advantage(trajectories, config.step_gamma, config.step_lam)
    elif config.step_level_adv_estimator == StepLevelAdvantageEstimator.REINFORCE_PLUS_PLUS:
        return bi_level_core_algos.step_level_reinforce_pp_advantage(trajectories, config.step_gamma)
    else:
        raise NotImplementedError(
            f"Step-level advantage estimator {config.step_level_adv_estimator} is not implemented."
        )


def compute_bi_level_advantage(data: DataProto, config: ConfigDataclass):
    """Compute advantage for Bi-level RL, step-level and token-level.

    Args:
        data: `(DataProto)`
            shape: (bs)
        config: `(ConfigDataclass)`
            config of algorithm
    """
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)

    trajectories = collect_trajectory_data(data, FIELDS_TO_COLLECT)

    step_level_advantages, step_level_returns = compute_step_level_advantage(trajectories, config)

    # assign step-level return as the reward for the last token in each step
    for traj_index, traj in enumerate(trajectories):
        num_step = len(traj["step_level_rewards"])
        for step_index in range(num_step):
            data_index = traj["index_in_data"][step_index]
            valid_length = traj["valid_lengths"][step_index]
            data.batch["token_level_rewards"][data_index, valid_length - 1] = step_level_returns[traj_index][step_index]
    if config.token_level_adv_estimator == TokenLevelAdvantageEstimator.TRIVIAL:
        step_level_advantages = step_level_mask_whiten(step_level_advantages)
        token_level_advantages, token_level_returns = bi_level_core_algos.token_level_trivial_advantage(
            trajectories, step_level_advantages, step_level_returns, data.batch["response_mask"]
        )
    elif config.token_level_adv_estimator == TokenLevelAdvantageEstimator.GAE:
        token_level_advantages, token_level_returns = compute_gae_advantage_return(
            data.batch["token_level_rewards"],
            data.batch["token_level_values"],
            data.batch["response_mask"],
            config.token_gamma,
            config.token_lam,
        )
    else:
        raise NotImplementedError(
            f"Token-level advantage estimator {config.token_level_adv_estimator} is not implemented."
        )

    data.batch["advantages"] = token_level_advantages
    data.batch["returns"] = token_level_returns

    return data
