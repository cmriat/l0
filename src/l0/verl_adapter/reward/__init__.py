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
"""Reward function module."""

from typing import Any

from verl import DataProto

from .nb_agent_reward import (
    combine_str_reward_fn,
)

REWARD_FN = {
    "2WikiMultihopQA": combine_str_reward_fn,
    "natural_questions": combine_str_reward_fn,
    "hotpot_qa/distractor": combine_str_reward_fn,
    "hotpot_qa/fullwiki": combine_str_reward_fn,
    "musique": combine_str_reward_fn,
    "pop_qa": combine_str_reward_fn,
    "simple_qa": combine_str_reward_fn,
    "bamboogle": combine_str_reward_fn,
    "trivia_qa/rc.nocontext": combine_str_reward_fn,
    "trivia_qa/rc": combine_str_reward_fn,
    "2WikiMultihopQA": combine_str_reward_fn,
    "trivia_qa/unfiltered.nocontext": combine_str_reward_fn,
    "trivia_qa/unfiltered": combine_str_reward_fn,
    "trivia_qa/rc.web": combine_str_reward_fn,
    "trivia_qa/rc.web.nocontext": combine_str_reward_fn,
    "trivia_qa/rc.wikipedia.nocontext": combine_str_reward_fn,
    "trivia_qa/rc.wikipedia": combine_str_reward_fn,
    "hotpot_qa/distractor": combine_str_reward_fn,
    "hotpot_qa/fullwiki": combine_str_reward_fn,
    "musique": combine_str_reward_fn,
    "natural_questions": combine_str_reward_fn,
    "deepmath": combine_str_reward_fn,
}


def multi_step_reward_fn(
    data_source: str, data: DataProto, trajectory: dict
) -> tuple[list[float] | dict[str, list[Any]], dict]:
    """Dispatch function to compute the reward score based on the data source.

    Args:
        data_source: The data source.
        trajectory: The trajectory data containing actions and other relevant information.

    Returns:
        The computed score as a list of integers or a dictionary with additional information.
    """
    if data_source in REWARD_FN:
        reward_fn = REWARD_FN[data_source]
        return reward_fn(data, trajectory)
    else:
        raise ValueError(f"Unknown data source: {data_source}")


def load_reward_manager(config, tokenizer, num_examine, **reward_kwargs):
    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    if reward_manager_name == "multistep":
        from .multistep_reward_manager import MultistepRewardManager

        reward_manager_cls = MultistepRewardManager
    else:
        raise NotImplementedError

    compute_score = None
    return reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=compute_score,
        reward_fn_key=config.data.reward_fn_key,
        **reward_kwargs,
    )
