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

import os
import copy
import pickle
import logging
import multiprocessing as mp
from collections import Counter

from verl import DataProto
from verl.utils import hf_processor, hf_tokenizer
from config_dataclass import ConfigDataclass

from l0.traj_sampler import TrajectorySampler
from l0.traj_sampler.nb_agent_sampler.cmd_factory import get_run_nb_agent_worker_cmd
from l0.traj_sampler.nb_agent_sampler.worker_config import NBAgentWorkerConfig

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class NBAgentEvalSampler(TrajectorySampler):
    def __init__(self, config: ConfigDataclass):
        super().__init__(config)
        self.config = config
        self.tokenizer = hf_tokenizer(self.config.model_id, trust_remote_code=True)
        self.processor = hf_processor(self.config.model_id, use_fast=True)

    def _traj_meta_info_to_step_meta_info(self, meta_info: dict, num_steps: int) -> list[dict]:
        """Convert meta information from trajectory to step meta information."""
        step_meta_infos = []
        for i in range(num_steps):
            if i == num_steps - 1:
                step_meta_infos.append(meta_info)
            else:
                step_meta_infos.append({key: None for key in meta_info.keys()})
        return step_meta_infos

    def pre_process(self, batch: DataProto, eval_result_dir: str) -> list[ConfigDataclass]:
        configs = []
        traj_id_counter = Counter()

        for i in range(len(batch.non_tensor_batch["question"])):
            config_dict = copy.deepcopy(self.config.agent)
            config_dict.task_run.task = batch.non_tensor_batch["question"][i]
            config_dict.task_run.file_path = batch.non_tensor_batch["file_paths"][i]
            config_dict.uuid = (
                batch.non_tensor_batch["uuids"][i] + "_" + str(traj_id_counter[batch.non_tensor_batch["uuids"][i]])
            )
            config_dict.nb_agent.uuid = config_dict.uuid
            config_dict.nb_agent.traj_save_storage_path = os.path.join(
                eval_result_dir, config_dict.nb_agent.traj_save_storage_path
            )
            config_dict.log_path = os.path.join(eval_result_dir, config_dict.log_path)
            config = NBAgentWorkerConfig.from_dictconfig(config_dict)
            configs.append(config)
            traj_id_counter[batch.non_tensor_batch["uuids"][i]] += 1

        return configs

    def post_process(self, config: type[ConfigDataclass]) -> tuple[dict[str, list]]:
        """Post-process the trajectory data from the agent worker.

        Args:
            config: Configuration object for the agent worker

        Returns:
            Processed tensor and non-tensor trajectory data
        """
        traj_path = config.nb_agent.traj_save_storage_path / config.uuid
        task_uuid = config.uuid.split("_")[0]
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
            try:
                # extract final answer from the last step
                final_answer = traj_data["meta"].get("final_answer")
            except Exception as e:
                logger.error(f"failed to extract final answer from {traj_data}, error: {e}")

        return task_uuid, final_answer

    def create_running_cmd(self, config: ConfigDataclass) -> list[str]:
        return get_run_nb_agent_worker_cmd(config)

    def batch_inference(
        self, batch: DataProto, minimal_batch_size: int | None = None, eval_result_dir: str | None = None
    ) -> dict[str, str]:
        batch_configs = self.pre_process(batch, eval_result_dir)

        if not batch_configs:
            raise ValueError("No valid configurations to process. Check the input batch.")

        if minimal_batch_size is None:
            minimal_batch_size = len(batch_configs)
        elif minimal_batch_size > len(batch_configs):
            logger.warning(
                f"Minimal batch size ({minimal_batch_size}) is greater than total tasks ({len(batch_configs)}). Clamping to total tasks."
            )
            minimal_batch_size = len(batch_configs)

        batched_configs = {config.uuid + f"_{i}": config for i, config in enumerate(batch_configs)}

        completed_queue = mp.Queue()
        terminate_event = mp.Event()

        producer = mp.Process(target=self._execute_batch, args=(batched_configs, completed_queue, terminate_event))
        producer.start()

        collected_count = 0
        final_answers = {}
        while True:
            sample_id = completed_queue.get()
            if sample_id is None:
                break
            try:
                task_uuid, final_answer = self.post_process(batched_configs[sample_id])
                final_answers[task_uuid] = final_answer
                collected_count += 1
                logger.debug(f"Processed trajectory {sample_id}, total: {collected_count}")
            except Exception as e:
                logger.error(f"Error processing trajectory {sample_id}: {e}")
            if collected_count >= minimal_batch_size:
                terminate_event.set()
                break

        logger.info("Waiting for producer process to join...")
        producer.join(timeout=max(10.0, self.task_timeout / 10))
        if producer.is_alive():
            logger.warning("Producer process did not join in time. Terminating it forcefully.")
            producer.terminate()
            producer.join(timeout=5.0)

        return final_answers

    def _get_sampled_data_str(self, tensor_data: dict[str, list], non_tensor_data: dict[str, list]) -> str:  # noqa: ARG002
        return ""
