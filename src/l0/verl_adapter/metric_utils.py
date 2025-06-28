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
"""Compute metrics for the PPO data batch."""

from typing import Any, Dict

import numpy as np
import torch
from verl import DataProto
from verl.trainer.ppo.metric_utils import _compute_response_info


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> Dict[str, Any]:
    """Compute metrics for the batch."""
    sequence_token_score = batch.batch["token_level_scores"].sum(-1)
    sequence_token_reward = batch.batch["token_level_rewards"].sum(-1)
    sequence_step_score = batch.batch["step_level_scores"]
    sequence_step_reward = batch.batch["step_level_rewards"]

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["token_level_values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    num_traj = len(np.unique(batch.non_tensor_batch["uuids"]))
    num_steps = len(batch)

    metrics = {
        # score
        "score/token_level/mean": torch.mean(sequence_token_score).detach().item(),
        "score/token_level/max": torch.max(sequence_token_score).detach().item(),
        "score/token_level/min": torch.min(sequence_token_score).detach().item(),
        "score/token_level/std": torch.std(sequence_token_score).detach().item(),
        "score/step_level/mean": torch.mean(sequence_step_score).detach().item(),
        "score/step_level/max": torch.max(sequence_step_score).detach().item(),
        "score/step_level/min": torch.min(sequence_step_score).detach().item(),
        "score/step_level/std": torch.std(sequence_step_score).detach().item(),
        # reward
        "rewards/token_level/mean": torch.mean(sequence_token_reward).detach().item(),
        "rewards/token_level/max": torch.max(sequence_token_reward).detach().item(),
        "rewards/token_level/min": torch.min(sequence_token_reward).detach().item(),
        "rewards/token_level/std": torch.std(sequence_token_reward).detach().item(),
        "rewards/step_level/mean": torch.mean(sequence_step_reward).detach().item(),
        "rewards/step_level/max": torch.max(sequence_step_reward).detach().item(),
        "rewards/step_level/min": torch.min(sequence_step_reward).detach().item(),
        "rewards/step_level/std": torch.std(sequence_step_reward).detach().item(),
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        "critic/returns/std": torch.std(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                "critic/values/std": torch.std(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/std": torch.std(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float())
        .detach()
        .item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/std": torch.std(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
        # traj and step counts
        "sampler/traj_num": num_traj,
        "sampler/step_num": num_steps,
    }
    return metrics
