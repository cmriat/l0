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
"""Trajectory sampler for NB agent."""

import copy
import pickle

from verl import DataProto
from verl.utils import hf_processor, hf_tokenizer
from config_dataclass import ConfigDataclass
from nbagent.nb_basic import NBAgentStep

from l0.traj_sampler import TrajectorySampler

from .cmd_factory import get_run_nb_agent_worker_cmd
from .worker_config import NBAgentWorkerConfig


class NBAgentSampler(TrajectorySampler):
    def __init__(self, config: ConfigDataclass):
        super().__init__(config)
        self.config = config
        self.tokenizer = hf_tokenizer(self.config.model_id, trust_remote_code=True)
        self.processor = hf_processor(self.config.model_id, use_fast=True)

    def convert_step_to_token_ids(self, step: NBAgentStep) -> dict[str, list[int]]:
        """Convert an AgentStep to token IDs for both prompt and response.

        Args:
            step: The agent step to convert

        Returns:
            Dict containing prompt and response token IDs
        """
        # Convert state messages to prompt
        prompt_messages = []
        for msg in step.state:
            if isinstance(msg["content"], str):
                prompt_messages.append({"role": msg["role"], "content": msg["content"]})
            else:
                content_list = msg["content"]
                for content in content_list:
                    if content["type"] == "text":
                        prompt_messages.append({"role": msg["role"], "content": content["text"]})

        model_inputs = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            add_special_tokens=False,
            truncation=True,
            padding="max_length",
            padding_side="left",
            max_length=self.config.agent.nb_agent.max_tokens - self.config.agent.nb_agent.max_response_length,
            return_attention_mask=True,
        )
        prompts_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        response = step.action.full_response
        response_output = self.tokenizer(
            response,
            add_special_tokens=False,
            truncation=True,
            padding="max_length",
            max_length=self.config.agent.nb_agent.max_response_length,
        )
        response_ids = response_output["input_ids"]
        input_ids = prompts_ids + response_ids
        attention_mask = attention_mask + response_output["attention_mask"]
        return {
            "prompts": prompts_ids,
            "input_ids": input_ids,
            "response_ids": response_ids,
            "attention_mask": attention_mask,
        }

    def _traj_meta_info_to_step_meta_info(self, meta_info: dict, num_steps: int) -> list[dict]:
        """Convert meta information from trajectory to step meta information."""
        step_meta_infos = []
        for i in range(num_steps):
            if i == num_steps - 1:
                step_meta_infos.append(meta_info)
            else:
                step_meta_infos.append({key: None for key in meta_info.keys()})
        return step_meta_infos

    def pre_process(self, batch: DataProto) -> list[ConfigDataclass]:
        configs = []

        for i in range(len(batch.non_tensor_batch["question"])):
            config_dict = copy.deepcopy(self.config.agent)
            config_dict.task_run.task = batch.non_tensor_batch["question"][i]
            config_dict.uuid = batch.non_tensor_batch["uuids"][i]
            config_dict.nb_agent.uuid = config_dict.uuid
            config = NBAgentWorkerConfig.from_dictconfig(config_dict)
            config.nb_agent.tool_ability = batch.non_tensor_batch["ability"][i]
            configs.append(config)

        return configs

    def post_process(self, config: type[ConfigDataclass]) -> tuple[dict[str, list]]:
        """Post-process the trajectory data from the agent worker.

        Args:
            config: Configuration object for the agent worker

        Returns:
            Processed tensor and non-tensor trajectory data
        """
        traj_path = config.nb_agent.traj_save_storage_path / config.uuid

        # Initialize lists to collect all token IDs
        tensor_data = {
            "max_prompt_length": [],
            "prompts": [],
            "input_ids": [],
            "responses": [],
            "attention_mask": [],
        }
        non_tensor_data = {
            "nb_steps": [],
            "meta_info": [],
        }

        # Ideally, there should be only one file
        traj_paths = list(traj_path.glob("*.pkl"))
        if not traj_paths:
            raise ValueError(f"No trajectory file found in {traj_path}")
        if len(traj_paths) > 1:
            raise ValueError(
                f"Multiple trajectory files found ({len(traj_paths)}) in {traj_path}, expected exactly one"
            )
        traj_file = traj_paths[0]

        with traj_file.open("rb") as f:
            traj_data = pickle.load(f)
            traj: list[NBAgentStep] = traj_data["trajectories"]
            non_tensor_data["meta_info"].extend(self._traj_meta_info_to_step_meta_info(traj_data["meta"], len(traj)))

            # Process each step in the trajectory
            for step in traj:
                # Convert step to token IDs
                token_ids = self.convert_step_to_token_ids(step)
                tensor_data["prompts"].append(token_ids["prompts"])
                tensor_data["input_ids"].append(token_ids["input_ids"])
                tensor_data["responses"].append(token_ids["response_ids"])
                tensor_data["attention_mask"].append(token_ids["attention_mask"])
                tensor_data["max_prompt_length"].append(
                    self.config.agent.nb_agent.max_tokens - self.config.agent.nb_agent.max_response_length
                )
                non_tensor_data["nb_steps"].append(step)

        return tensor_data, non_tensor_data

    def create_running_cmd(self, config: ConfigDataclass) -> list[str]:
        return get_run_nb_agent_worker_cmd(config)

    def _get_sampled_data_str(self, tensor_data: dict[str, list], non_tensor_data: dict[str, list]) -> str:  # noqa: ARG002
        formatted_str = ""
        state = non_tensor_data["nb_steps"][-1].state
        for message in state:
            formatted_str += f"{message['role'].capitalize()}\n: {message['content'][0]['text']}\n"
        return formatted_str
