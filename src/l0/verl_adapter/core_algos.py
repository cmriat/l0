# Copyright (c) 2022–2025 China Merchants Research Institute of Advanced Technology Corporation and its Affiliates
#
# This file is adapted from the verl library.
# Copyright 2023-2024 Bytedance Ltd. and/or its affiliates.
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
#
# NOTICE: This file has been modified by China Merchants Research Institute Of Advanced Technology from its original version.

"""Core functions to implement PPO algorithms."""

import torch


def step_level_gae_advantage(trajectories, gamma, lam) -> tuple[list[list[float]], list[list[float]]]:
    """Compute step-level GAE advantage and returns."""
    advantages = []
    returns = []
    for traj in trajectories:
        num_steps = len(traj["step_level_values"])
        lastgaelam = 0
        traj_advantages = []
        traj_returns = []
        for t in reversed(range(num_steps)):
            nextvalues = traj["step_level_values"][t + 1] if t < num_steps - 1 else 0.0
            delta = traj["step_level_rewards"][t] + gamma * nextvalues - traj["step_level_values"][t]
            lastgaelam = delta + gamma * lam * lastgaelam
            traj_advantages.append(lastgaelam)
            traj_returns.append(lastgaelam + traj["step_level_values"][t])
        traj_advantages.reverse()
        traj_returns.reverse()

        advantages.append(traj_advantages)
        returns.append(traj_returns)
    return advantages, returns


def step_level_reinforce_pp_advantage(trajectories, gamma) -> tuple[list[list[float]], list[list[float]]]:
    """Compute reinforce++ advantage and returns."""
    advantages = []
    returns = []
    for traj in trajectories:
        num_steps = len(traj["step_level_rewards"])
        running_return = 0
        traj_advantages = []
        traj_returns = []
        for t in reversed(range(num_steps)):
            running_return = traj["step_level_rewards"][t] + gamma * running_return
            traj_returns.append(running_return)
            traj_advantages.append(running_return)
        traj_advantages.reverse()
        traj_returns.reverse()

        advantages.append(traj_advantages)
        returns.append(traj_returns)
    return advantages, returns


def token_level_trivial_advantage(
    trajectories, step_advantages, step_returns, response_mask
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute token-level trivial advantage."""
    returns = torch.zeros_like(response_mask, dtype=torch.float32)
    advantages = torch.zeros_like(response_mask, dtype=torch.float32)
    for traj_index, traj in enumerate(trajectories):
        num_steps = len(step_advantages[traj_index])
        for step_index in range(num_steps):
            data_index = traj["index_in_data"][step_index]
            advantages[data_index, :] = step_advantages[traj_index][step_index]
            returns[data_index, :] = step_returns[traj_index][step_index]
    # TODO: should we whiten the advantages in token-level?
    # advantages = masked_whiten(advantages, response_mask)
    return advantages, returns
