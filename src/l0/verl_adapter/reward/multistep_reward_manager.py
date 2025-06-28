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
"""Multistep reward manager for agent reinforcement learning."""

from collections import defaultdict

import numpy as np
import torch
from verl import DataProto

from l0.utils import collect_trajectory_data
from l0.verl_adapter.reward import multi_step_reward_fn


class MultistepRewardManager:
    """Multistep reward manager for agent reinforcement learning.

    This class is used to manage the rewards for multiple steps in a reinforcement learning environment.
    It allows for the computation of rewards based on the current and previous states, actions, and rewards.
    """

    def __init__(self, tokenizer, num_examine, compute_score, reward_fn_key="data_source"):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        if compute_score is not None:
            raise NotImplementedError
        else:
            self.compute_score = multi_step_reward_fn
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict=False) -> tuple[torch.Tensor, torch.Tensor] | dict:
        """Compute rewards for the given data."""
        # Collect trajectory data
        trajectories = self._group_trajectories(data)

        # Compute rewards
        step_level_reward_tensor, token_level_reward_tensor, reward_extra_info, metrics, traj_score_info = (
            self._compute_rewards(data, trajectories)
        )

        # Return results
        if return_dict:
            return {
                "step_level_reward_tensor": step_level_reward_tensor,
                "token_level_reward_tensor": token_level_reward_tensor,
                "reward_extra_info": reward_extra_info,
                "metrics": metrics,
                "traj_score_info": traj_score_info,
            }
        return step_level_reward_tensor, token_level_reward_tensor

    def _group_trajectories(self, data: DataProto) -> list[dict]:
        """Collect data for all trajectories."""

        def extract_action(step):
            response_ids = step.batch["responses"]
            valid_response_length = step.batch["attention_mask"][step.batch["max_prompt_length"] :].sum()
            valid_response_ids = response_ids[:valid_response_length]
            return self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

        def extract_ground_truth(step):
            return step.non_tensor_batch["reward_model"]["ground_truth"]

        def extract_data_source(step):
            return step.non_tensor_batch[self.reward_fn_key]

        def extract_question(step):
            return step.non_tensor_batch["question"]

        fields_to_collect = {
            "actions": extract_action,
            "ground_truth": extract_ground_truth,
            "data_source": extract_data_source,
            "question": extract_question,
        }

        return collect_trajectory_data(data, fields_to_collect)

    def _compute_rewards(
        self, data: DataProto, trajectories: list[dict]
    ) -> tuple[torch.Tensor, torch.Tensor, dict, dict]:
        """Compute rewards for each step."""
        step_level_reward_tensor = torch.zeros(data.batch["responses"].shape[0], dtype=torch.float32)
        token_level_reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        sample_num = data.batch["responses"].shape[0]
        reward_extra_info = {}
        raw_metrics = defaultdict(list)
        traj_score_info = defaultdict(list)

        for trajectory in trajectories:
            raw_reward_extra_info = defaultdict(list)
            scores, traj_metrics = self.compute_score(
                data_source=trajectory["data_source"][0],
                data=data,
                trajectory=trajectory,
            )

            traj_score_info["data_source"].append(trajectory["data_source"][0])
            traj_score_info["question"].append(trajectory["question"][0])

            for k, v in traj_metrics.items():
                if isinstance(v, list):
                    raw_metrics[k].extend(v)
                elif isinstance(v, (int, float)):
                    raw_metrics[k].append(float(v))
                    traj_score_info[k].append(float(v))

            if isinstance(scores, dict):
                for k, v in scores.items():
                    if k == "score":
                        continue
                    raw_reward_extra_info[k] = v
                scores = scores["score"]

            for i, score in enumerate(scores):
                original_data_index = trajectory["index_in_data"][i]
                valid_length = trajectory["valid_lengths"][i]
                step_level_reward_tensor[original_data_index] = score
                token_level_reward_tensor[original_data_index, valid_length - 1] = score

                for key in raw_reward_extra_info.keys():
                    if key not in reward_extra_info:
                        reward_extra_info[key] = [None] * sample_num
                    reward_extra_info[key][original_data_index] = raw_reward_extra_info[key][i]

        # calculate mean, std, and max for raw metrics
        metrics = {}
        for k, v in raw_metrics.items():
            metrics[f"score/{k}_mean"] = np.mean(v)
            metrics[f"score/{k}_std"] = np.std(v)
            metrics[f"score/{k}_max"] = np.max(v)
            metrics[f"score/{k}_min"] = np.min(v)

        return step_level_reward_tensor, token_level_reward_tensor, reward_extra_info, metrics, traj_score_info
