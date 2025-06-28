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
"""Test the nb agent worker."""

import logging
import tempfile
import subprocess
from uuid import uuid4
from datetime import datetime

import dotenv
from upath import UPath

from l0.traj_sampler.nb_agent_sampler import (
    NBAgentWorkerConfig,
    get_run_nb_agent_worker_cmd,
)

dotenv.load_dotenv(override=True)


def test_run_nb_agent_in_bwrap():
    """Test running the nb agent in a bubblewrap sandbox.

    This test verifies that:
    1. A nb agent can be launched in a bwrap sandbox
    2. The agent successfully completes a simple task
    3. The agent saves trajectories to the specified path
    4. The agent generates appropriate log files
    """
    with tempfile.TemporaryDirectory() as traj_save_dir, tempfile.TemporaryDirectory() as log_dir:
        traj_save_path = UPath(traj_save_dir)
        log_path = UPath(log_dir)

        test_uuid = str(uuid4())
        test_task = "What is 2+2? Provide the answer directly."
        config_dict = {
            "uuid": test_uuid,
            "task_run": {
                "max_steps": 2,
                "task": test_task,
            },
            "openai_server_model": {
                "model_id": "google/gemini-2.5-flash-preview-05-20",
                "server_mode": "default",
            },
            "nb_agent": {
                "uuid": test_uuid,
                "max_tokens": 10000,
                "max_response_length": 3192,
                "traj_save_storage_path": str(traj_save_path),
                "agent_mode": "llm",
                "sampling_params": {
                    "top_p": 0.95,
                },
                "tool_ability": "empty",
            },
            "log_path": str(log_path),
        }

        config = NBAgentWorkerConfig.from_dict(config_dict)

        bwrap_cmd = get_run_nb_agent_worker_cmd(config)

        # Execute the command in a subprocess
        logging.info(f"Running nb agent in bwrap with UUID: {test_uuid}")
        process = subprocess.Popen(
            bwrap_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = process.communicate()

        # Print any output for debugging
        if stdout:
            logging.info("STDOUT: %s", stdout)
        if stderr:
            logging.info("STDERR: %s", stderr)

        # Check if the process ran successfully
        assert process.returncode == 0, f"Process failed with return code {process.returncode}"

        # Check if trajectory files were created
        traj_files = list(traj_save_path.glob(f"{test_uuid}/*"))
        assert len(traj_files) > 0, "No trajectory files were created"

        # Check if log files were created
        # Get current date in the format used by Hydra
        current_date = datetime.now().strftime("%Y-%m-%d")
        # Look for log directories created in the last minute
        log_dirs = list(log_path.glob(f"*_{current_date}_*"))
        assert len(log_dirs) > 0, "No log directory was created"

        # Check for specific log files in the most recent log directory
        latest_log_dir = max(log_dirs, key=lambda x: x.stat().st_mtime)
        log_files = list(latest_log_dir.glob("*.log"))
        assert len(log_files) > 0, "No log files were created"

        # Check if there's a hydra config file
        config_files = list(latest_log_dir.glob(".hydra/config.yaml"))
        assert len(config_files) > 0, "No Hydra config file was created"

        logging.info("Test passed. Trajectory files created: %s", [str(f) for f in traj_files])
        logging.info("Log files created: %s", [str(f) for f in log_files])


if __name__ == "__main__":
    test_run_nb_agent_in_bwrap()
