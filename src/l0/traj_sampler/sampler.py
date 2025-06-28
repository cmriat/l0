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
"""Sampler for agent trajectories."""

import multiprocessing as mp

mp.set_start_method("spawn", force=True)

import os
import time
import logging
import subprocess
import multiprocessing as mp
from abc import ABC, abstractmethod
from enum import Enum
from collections import deque

import httpx
import torch
from verl import DataProto
from upath import UPath
from config_dataclass import ConfigDataclass
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class TaskStatus(Enum):
    """Enum for task status."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class TrajectorySampler(ABC):
    """Base class for trajectory samplers."""

    def __init__(self, config: ConfigDataclass):
        self.config = config
        self.max_concurrency = config.max_concurrency
        self.max_retries = config.max_retries
        self.task_timeout = config.task_timeout

        logger.info(f"Sampling concurrency set to {self.max_concurrency}.")

    @abstractmethod
    def pre_process(self, batch: DataProto) -> list[ConfigDataclass]:
        """Pre-process the input batch to create configurations for agent workers."""
        pass

    @abstractmethod
    def create_running_cmd(self, config: ConfigDataclass) -> list[str]:
        """Create the command to run the agent worker process."""
        pass

    @abstractmethod
    def post_process(self, config: ConfigDataclass) -> tuple[dict[str, list], dict[str, list]]:
        """Post-process the trajectory data from the agent worker."""
        pass

    def _launch_worker_locally(self, config: ConfigDataclass, uuid: str) -> tuple[subprocess.Popen, str] | None:
        """Launch an agent worker synchronously and return the Popen object and its UUID."""
        try:
            cmd = self.create_running_cmd(config)
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            logger.debug(f"Launched worker {uuid} (PID: {proc.pid})")
            return proc, uuid
        except Exception as e:
            logger.debug(f"Exception launching {uuid}: {e}")
            return None

    def _launch_worker_remotely(self, config: ConfigDataclass, uuid: str) -> str | None:
        """Launch an agent worker remotely and return the UUID of the launched worker."""
        sample_id = int(uuid.split("_")[2])
        remote_exec_server_url = self.config.remote_exec_server_url[sample_id % len(self.config.remote_exec_server_url)]
        try:
            cmd = self.create_running_cmd(config)
            # round-robin the remote execution server URL based on sample_id
            response = httpx.post(
                f"{remote_exec_server_url}/create_task",
                json={
                    "cmd": cmd,
                    "uuid": uuid,
                },
                timeout=self.task_timeout,
            )
            response.raise_for_status()
            logger.info(f"Launched remote worker {uuid} on {remote_exec_server_url}")
            return remote_exec_server_url, uuid
        except Exception as e:
            logger.info(f"Exception launching remote worker {uuid} on {remote_exec_server_url}: {e}")
            return None

    def _launch_worker(self, config: ConfigDataclass, uuid: str) -> tuple[subprocess.Popen | UPath, str] | None:
        """Launch an agent worker either locally or remotely based on the executor configuration."""
        if self.config.executor_type == "remote":
            return self._launch_worker_remotely(config, uuid)
        elif self.config.executor_type == "local":
            return self._launch_worker_locally(config, uuid)
        else:
            raise ValueError(
                f"Unsupported executor type: {self.config.executor_type}. Supported types are 'remote' and 'local'."
            )

    def _execute_batch(
        self, batched_configs: dict[str, ConfigDataclass], completed_queue: mp.Queue, terminate_event: mp.Event
    ):
        """Execute a batch of agent worker tasks synchronously by polling subprocesses."""
        active_tasks: dict[
            str, tuple[subprocess.Popen | str, float, ConfigDataclass, int]
        ] = {}  # uuid: (handler, start_time, config, attempt_count)
        pending_tasks = deque(
            (uuid, config, 0) for (uuid, config) in batched_configs.items()
        )  # (uuid, config, attempt_count)

        try:
            while pending_tasks or active_tasks:
                if terminate_event.is_set():
                    logger.info("Terminate event set. Killing active worker processes.")
                    for uuid_key, (handler, _, _, _) in list(active_tasks.items()):
                        self._close_task(uuid_key, handler)
                        if uuid_key in active_tasks:
                            del active_tasks[uuid_key]
                    break

                while len(active_tasks) < self.max_concurrency and pending_tasks:
                    if terminate_event.is_set():
                        break

                    uuid, config_obj, attempt_count = pending_tasks.popleft()

                    if attempt_count >= self.max_retries:
                        logger.error(f"Worker {uuid} exceeded max retries ({self.max_retries}). Skipping.")
                        continue
                    launch_result = self._launch_worker(config_obj, uuid)
                    if launch_result is not None:
                        handler, launched_uuid = launch_result
                        active_tasks[launched_uuid] = (handler, time.time(), config_obj, attempt_count)
                    else:
                        pending_tasks.append((uuid, config_obj, attempt_count + 1))

                completed_uuids_this_iteration = []
                for s_uuid, (handler, start_time, config, attempt_count) in list(active_tasks.items()):
                    task_status = self._check_task_status(s_uuid, handler, start_time)
                    if task_status == TaskStatus.COMPLETED:
                        completed_uuids_this_iteration.append(s_uuid)
                        completed_queue.put(s_uuid)
                    elif task_status == TaskStatus.TIMEOUT or task_status == TaskStatus.FAILED:
                        if task_status == TaskStatus.TIMEOUT:
                            logger.info(f"Worker {s_uuid} timed out after {self.task_timeout:.2f}s. Closing.")
                            self._close_task(s_uuid, handler)
                        completed_uuids_this_iteration.append(s_uuid)
                        pending_tasks.append((s_uuid, config, attempt_count + 1))

                for uuid_to_remove in completed_uuids_this_iteration:
                    if uuid_to_remove in active_tasks:
                        del active_tasks[uuid_to_remove]

                if not pending_tasks and not active_tasks:
                    logger.info("All tasks processed or failed to launch.")
                    break

                time.sleep(0.5)

        except Exception as e:
            logger.error(f"Unexpected error in _execute_batch: {e}", exc_info=True)

        completed_queue.put(None)

    def _close_task_remotely(
        self, uuid: str, remote_url: str, timeout_graceful: float = 1.0, timeout_force: float = 1.0
    ):
        """Close the task handler remotely."""
        try:
            response = httpx.post(
                f"{remote_url}/close_task/{uuid}",
                json={"timeout_graceful": timeout_graceful, "timeout_force": timeout_force},
                timeout=timeout_graceful + timeout_force,
            )
            response.raise_for_status()
            logger.info(f"Closed remote task {uuid} successfully.")
        except httpx.RequestError as e:
            logger.error(f"Failed to close remote task {uuid}: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error while closing remote task {uuid}: {e}")

    def _close_task_locally(self, proc, timeout_graceful: float = 1.0, timeout_force: float = 1.0):
        """Attempts to terminate the process gracefully, then forcefully kills it if necessary.

        First sends SIGTERM, waits, then sends SIGKILL if still alive.
        """
        if proc.poll() is not None:
            logger.debug(f"Process {proc.pid} already terminated.")
            return
        try:
            logger.debug(f"Sending SIGTERM to process {proc.pid}.")
            proc.terminate()
            proc.wait(timeout=timeout_graceful)
            if proc.poll() is not None:
                logger.debug(f"Process {proc.pid} terminated gracefully after SIGTERM.")
                return
        except subprocess.TimeoutExpired:
            logger.warning(
                f"Process {proc.pid} did not terminate via SIGTERM within {timeout_graceful}s. Attempting SIGKILL."
            )
        except Exception as e:
            logger.debug(
                f"Exception during SIGTERM handling for process {proc.pid}: {e}. Checking if SIGKILL is needed."
            )

        if proc.poll() is None:
            logger.debug(f"Sending SIGKILL to process {proc.pid}.")
            try:
                proc.kill()
                proc.wait(timeout=timeout_force)
                if proc.poll() is not None:
                    logger.debug(f"Process {proc.pid} terminated after SIGKILL.")
                else:
                    logger.error(
                        f"Process {proc.pid} still running after SIGKILL and wait (no timeout). This is unexpected."
                    )
            except subprocess.TimeoutExpired:
                logger.error(
                    f"Process {proc.pid} failed to die via SIGKILL within {timeout_force}s. It might be a zombie or in an uninterruptible state."
                )
            except Exception as e:
                logger.debug(f"Exception during SIGKILL handling for process {proc.pid}: {e}.")
        else:
            logger.debug(f"Process {proc.pid} terminated before SIGKILL was attempted.")

    def _close_task(
        self,
        uuid: str,
        handler: subprocess.Popen | str,
        timeout_graceful: float = 1.0,
        timeout_force: float = 1.0,
    ):
        """Close the task handler gracefully, then forcefully if it does not terminate."""
        logger.info(f"Closing task {uuid}.")
        if self.config.executor_type == "remote":
            self._close_task_remotely(uuid, handler, timeout_graceful, timeout_force)
        else:
            self._close_task_locally(handler, timeout_graceful, timeout_force)

    def _check_task_status_locally(self, uuid: str, proc: subprocess.Popen) -> TaskStatus:
        if proc.poll() is not None:
            stdout_str, stderr_str = proc.communicate()
            if proc.returncode == 0:
                if stdout_str:
                    logger.debug(f"Agent {uuid} output: {stdout_str.strip()}")
                return TaskStatus.COMPLETED
            else:
                error_msg = stderr_str.strip() if stderr_str else "Unknown error"
                logger.debug(f"Worker {uuid} (PID: {proc.pid}) failed with code {proc.returncode}: {error_msg}")
                return TaskStatus.FAILED
        else:
            logger.debug(f"Worker {uuid} (PID: {proc.pid}) is still running.")
            return TaskStatus.RUNNING

    def _check_task_status_remotely(self, uuid: str, remote_url: str) -> TaskStatus:
        status = TaskStatus.RUNNING
        response = httpx.get(f"{remote_url}/check_task_status/{uuid}")
        if response.status_code == 200:
            status_data = response.json().get("status", "running")
            try:
                status = TaskStatus[status_data.upper()]
            except Exception:
                logger.warning(f"Unknown status '{status_data}' for worker {uuid}. Defaulting to RUNNING.")
        else:
            logger.warning(f"Failed to get status for worker {uuid} from remote server: {response.status_code}")
        return status

    def _check_task_status(self, uuid: str, handler: subprocess.Popen | str, start_time: float) -> TaskStatus:
        task_status = TaskStatus.RUNNING
        if self.config.executor_type == "remote":
            task_status = self._check_task_status_remotely(uuid, handler)

        elif self.config.executor_type == "local":
            task_status = self._check_task_status_locally(uuid, handler)

        if task_status == TaskStatus.RUNNING and time.time() - start_time > self.task_timeout:
            logger.warning(f"Worker {uuid} timed out after {self.task_timeout:.2f}s.")
            task_status = TaskStatus.TIMEOUT

        return task_status

    @abstractmethod
    def _get_sampled_data_str(self, tensor_data: dict[str, list], non_tensor_data: dict[str, list]) -> str:
        """Show sampled data."""
        pass

    def get_batched_trajectories(self, batch: DataProto, minimal_batch_size: int | None = None) -> DataProto:
        """Get trajectories from the sampler.

        Args:
            batch: Input batch data
            minimal_batch_size: Minimum number of trajectories to collect before returning

        Returns:
            Processed trajectory data
        """
        batch_configs = self.pre_process(batch)

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

        tensor_datas = {}
        non_tensor_datas = {"uuids": [], "step_indexes": []}
        collected_count = 0

        while True:
            sample_id = completed_queue.get()
            if sample_id is None:
                break

            try:
                tensor_data, non_tensor_data = self.post_process(batched_configs[sample_id])
                if not tensor_data:
                    continue
                self._extend_dict(tensor_datas, tensor_data)
                self._extend_dict(non_tensor_datas, non_tensor_data)
                data_uuid = batched_configs[sample_id].uuid
                num_steps = len(tensor_data["input_ids"])
                non_tensor_datas["uuids"].extend([data_uuid] * num_steps)
                non_tensor_datas["step_indexes"].extend(list(range(num_steps)))

                if collected_count <= 1:
                    logger.info(f"Sampled data for trajectory {sample_id}:")
                    logger.info(self._get_sampled_data_str(tensor_data, non_tensor_data))

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

        # Convert collected data to tensors
        for key in tensor_datas.keys():
            tensor_datas[key] = torch.tensor(tensor_datas[key])
        if not tensor_datas:
            return DataProto()
        position_ids = compute_position_id_with_mask(tensor_datas["attention_mask"])
        tensor_datas["position_ids"] = position_ids

        final_collected_steps = len(non_tensor_datas.get("uuids", []))
        metrics = {"sampler/total_trajs": collected_count, "sampler/total_steps": final_collected_steps}
        batch_data = DataProto.from_dict(
            tensors=tensor_datas,
            non_tensors=non_tensor_datas,
            meta_info={"metrics": metrics},
        )
        return batch_data

    def _gracefully_terminate(self, proc, timeout_graceful: float = 1.0, timeout_force: float = 1.0):
        """Attempts to terminate the process gracefully, then forcefully kills it if necessary.

        First sends SIGTERM, waits, then sends SIGKILL if still alive.
        """
        if proc.poll() is not None:
            logger.debug(f"Process {proc.pid} already terminated.")
            return
        try:
            logger.debug(f"Sending SIGTERM to process {proc.pid}.")
            proc.terminate()
            proc.wait(timeout=timeout_graceful)
            if proc.poll() is not None:
                logger.debug(f"Process {proc.pid} terminated gracefully after SIGTERM.")
                return
        except subprocess.TimeoutExpired:
            logger.debug(
                f"Process {proc.pid} did not terminate via SIGTERM within {timeout_graceful}s. Attempting SIGKILL."
            )
        except Exception as e:
            logger.debug(
                f"Exception during SIGTERM handling for process {proc.pid}: {e}. Checking if SIGKILL is needed."
            )

        if proc.poll() is None:
            logger.debug(f"Sending SIGKILL to process {proc.pid}.")
            try:
                proc.kill()
                proc.wait(timeout=timeout_force)
                if proc.poll() is not None:
                    logger.debug(f"Process {proc.pid} terminated after SIGKILL.")
                else:
                    logger.error(
                        f"Process {proc.pid} still running after SIGKILL and wait (no timeout). This is unexpected."
                    )
            except subprocess.TimeoutExpired:
                logger.error(
                    f"Process {proc.pid} failed to die via SIGKILL within {timeout_force}s. It might be a zombie or in an uninterruptible state."
                )
            except Exception as e:
                logger.debug(f"Exception during SIGKILL handling for process {proc.pid}: {e}.")
        else:
            logger.debug(f"Process {proc.pid} terminated before SIGKILL was attempted.")

    def _extend_dict(self, target_dict: dict[str, list], source_dict: dict[str, list]):
        """Extend a target dictionary with lists from a source dictionary."""
        for key, value in source_dict.items():
            if key not in target_dict:
                target_dict[key] = []
            target_dict[key].extend(value)
