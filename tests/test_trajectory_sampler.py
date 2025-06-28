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
"""Test the TrajectorySampler class."""

import multiprocessing as mp

mp.set_start_method("spawn", force=True)

import dotenv

dotenv.load_dotenv(override=True)

import os
import time
import logging
import tempfile
import subprocess
from uuid import uuid4

import torch
import pytest
from verl import DataProto
from omegaconf import OmegaConf, DictConfig
from transformers import AutoTokenizer

from l0.traj_sampler import TrajectorySampler
from l0.dataset.qa_dataset import QADataset
from l0.traj_sampler.nb_agent_sampler import NBAgentSampler

MODEL_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"
DATA_FILE = "/data/agent_datasets/qa_datasets/tmp_filtered/validation.parquet"


class DummySampler(TrajectorySampler):
    """A dummy sampler to mock the NBAgentSampler for parallelism testing."""

    def __init__(self, config, num_tasks=4, task_sleep_duration=1):
        super().__init__(config)
        self.num_tasks = num_tasks
        self.task_sleep_duration = task_sleep_duration

    def pre_process(self, batch):  # noqa: ARG002
        configs = [
            {
                "task": f"Task {i}",
                "uuid": "a_b",
            }
            for i in range(self.num_tasks)
        ]
        return [DictConfig(config) for config in configs]

    def create_running_cmd(self, config):  # noqa: ARG002
        return ["sh", "-c", f"sleep {self.task_sleep_duration}"]

    def post_process(self, batch):  # noqa: ARG002
        return ({"input_ids": [[0]], "attention_mask": torch.zeros(1, 1)}, {"mock_meta": ["data"]})

    def _get_sampled_data_str(self, tensor_data, non_tensor_data) -> str:  # noqa: ARG002
        return ""


@pytest.fixture
def config():
    config = OmegaConf.load("tests/config/ppo_trainer.yaml").actor_rollout_ref.traj_sampler
    config.agent.openai_server_model.base_url = os.getenv("OPENAI_API_BASE")
    config.agent.nb_agent.max_tokens = 4096
    config.agent.nb_agent.max_response_length = 256
    config.model_id = "Qwen/Qwen2.5-7B-Instruct"  # for tokenizer / pre-processor
    config.agent.openai_server_model.model_id = "qwen/qwen-2.5-7b-instruct"
    return config


def get_test_data(num_sample: int) -> DataProto:
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_ID)
    dict_config = DictConfig({"max_prompt_length": 100000, "prompt_key": "question"})
    qa_dataset = QADataset(
        tokenizer=tokenizer,
        data_files=[DATA_FILE],
        config=dict_config,
    )
    # Create test configurations with different tasks
    non_tensor_batch = {
        "uuids": [str(uuid4()) + "_0" for _ in range(num_sample)],
        "question": [qa_dataset[i]["question"] for i in range(num_sample)],
        "ability": ["qa"] * num_sample,
    }
    batch_data = DataProto.from_dict(
        tensors={"test": torch.zeros((num_sample, 10))},  # Dummy tensor data
        non_tensors=non_tensor_batch,
    )
    return batch_data


def test_trajectory_sampler_parallelism(config):
    """Test the parallelism of TrajectorySampler by mocking worker execution."""
    NUM_TASKS = 4
    MAX_WORKERS = 2
    TASK_SLEEP_DURATION = 5

    fixed_overhead = 8

    config.max_concurrency = MAX_WORKERS
    sampler = DummySampler(config, num_tasks=NUM_TASKS, task_sleep_duration=TASK_SLEEP_DURATION)

    start_time = time.time()
    # minimal_batch_size=NUM_TASKS ensures we wait for all tasks.
    _ = sampler.get_batched_trajectories(None, minimal_batch_size=NUM_TASKS)
    end_time = time.time()

    duration = end_time - start_time

    logging.info(
        f"Parallelism Test: Execution time for {NUM_TASKS} tasks ({TASK_SLEEP_DURATION}s each) "
        f"with {MAX_WORKERS} workers: {duration:.2f}s"
    )

    expected_parallel_duration = ((NUM_TASKS + MAX_WORKERS - 1) // MAX_WORKERS) * TASK_SLEEP_DURATION
    expected_sequential_duration = NUM_TASKS * TASK_SLEEP_DURATION

    logging.info(f"Expected parallel duration (ideal): {expected_parallel_duration:.2f}s")
    logging.info(f"Expected sequential duration (ideal): {expected_sequential_duration:.2f}s")

    # Assertions
    # 1. Duration should be greater than a single task's duration (plus small buffer) if multiple tasks run over fewer workers.
    if NUM_TASKS > 1 and MAX_WORKERS < NUM_TASKS:
        assert duration > TASK_SLEEP_DURATION * 0.9, (
            f"Duration {duration:.2f}s too short, tasks might not have run correctly."
        )

    # 2. Duration should be significantly less than sequential execution if parallelism is effective.
    if MAX_WORKERS > 1 and NUM_TASKS > MAX_WORKERS:
        assert duration < expected_sequential_duration * 0.9 + fixed_overhead, (
            f"Duration {duration:.2f}s ({expected_sequential_duration * 0.9:.2f}s limit) is too close to sequential time {expected_sequential_duration:.2f}s. Parallelism not evident."
        )

    overhead_allowance = (
        max(1.0, expected_parallel_duration * 0.3) + fixed_overhead
    )  # Allow 1s or 50% of expected time as overhead

    assert duration >= expected_parallel_duration * 0.9, (
        f"Duration {duration:.2f}s is less than expected parallel time {expected_parallel_duration:.2f}s (with 10% tolerance)."
    )
    assert duration < expected_parallel_duration + overhead_allowance, (
        f"Duration {duration:.2f}s exceeds expected parallel time {expected_parallel_duration:.2f}s by more than {overhead_allowance:.2f}s overhead allowance."
    )


def test_nb_agent_locally_trajectory_sampler(config):
    """Test locally sampling trajectories from multiple nb agents using TrajectorySampler."""
    MAX_STEP = 1
    BATCH_SIZE = 4
    MIN_BATCH_SIZE = 2

    with tempfile.TemporaryDirectory() as traj_save_dir, tempfile.TemporaryDirectory() as log_dir:
        config.agent.nb_agent.traj_save_storage_path = traj_save_dir
        config.agent.openai_server_model.server_mode = "default"
        config.agent.log_path = log_dir
        config.agent.task_run.max_steps = MAX_STEP
        config.max_concurrency = 4

        sampler = NBAgentSampler(config=config)

        batch_data = get_test_data(BATCH_SIZE)
        results = sampler.get_batched_trajectories(batch_data, minimal_batch_size=MIN_BATCH_SIZE)

        assert len(set(results.non_tensor_batch["uuids"])) >= MIN_BATCH_SIZE, (
            "Trajectory ids should have the same length as the number of trajectories"
        )


def test_nb_agent_remote_trajectory_sampler(config):
    """Test remotely sampling trajectories from multiple nb agents using TrajectorySampler."""
    MAX_STEP = 1
    BATCH_SIZE = 3
    MIN_BATCH_SIZE = 2
    with tempfile.TemporaryDirectory() as traj_save_dir, tempfile.TemporaryDirectory() as log_dir:
        proc = subprocess.Popen(
            ["python", "-m", "l0.traj_sampler.task_server"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2)

        # s3_path = UPath("s3://lionrock-algo")
        # traj_save_dir = s3_path / "trajectories"
        log_dir = log_dir

        config.agent.nb_agent.traj_save_storage_path = traj_save_dir
        config.agent.openai_server_model.server_mode = "default"
        config.remote_exec_server_url = ["http://127.0.0.1:8000"]
        config.executor_type = "remote"
        config.agent.log_path = log_dir
        config.agent.task_run.max_steps = MAX_STEP
        config.max_concurrency = 4

        sampler = NBAgentSampler(config=config)

        batch_data = get_test_data(BATCH_SIZE)

        results = sampler.get_batched_trajectories(batch_data, minimal_batch_size=MIN_BATCH_SIZE)

        proc.terminate()
        proc.wait()

        assert len(set(results.non_tensor_batch["uuids"])) >= MIN_BATCH_SIZE, (
            "Trajectory ids should have the same length as the number of trajectories"
        )


if __name__ == "__main__":
    test_nb_agent_remote_trajectory_sampler(config)
