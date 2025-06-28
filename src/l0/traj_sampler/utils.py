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
"""Utility functions for for trajectory sampler."""

import os
import logging
from enum import Enum

from upath import UPath

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class BindPermission(Enum):
    R = "--ro-bind"
    RW = "--bind"


def format_bwrap_cmd(
    bind_paths: dict[UPath | str, tuple[UPath | str, str]],
    new_dirs: list[UPath | str],
    env_vars: dict[str, str],
    shell_cmd: str,
) -> list[str]:
    """Format the bwrap command.

    Args:
        bind_paths: A dictionary mapping paths to bind paths and permissions.
        new_dirs: A list of directories to create.
        env_vars: A dictionary mapping environment variables to values.
        shell_cmd: The shell command to run inside the sandbox.
    """
    logger.debug("*" * 10 + "Bwrap command" + "*" * 10)
    for path, (bind_path, permission) in bind_paths.items():
        logger.debug(f"{path} -> {bind_path} ({permission.value})")
    for dir_name in new_dirs:
        logger.debug(f"New dir: {dir_name}")
    for env_var, value in env_vars.items():
        logger.debug(f"{env_var} -> {value}")
    logger.debug("*" * 10 + "Bwrap command" + "*" * 10)

    bwrap_cmd = ["bwrap"]

    for path, (bind_path, permission) in bind_paths.items():
        if not isinstance(permission, BindPermission):
            raise ValueError(f"Invalid permission: {permission}")
        if isinstance(path, UPath):
            path = path.resolve()
        bwrap_cmd.extend([permission.value, str(path), str(bind_path)])

    for dir_name in new_dirs:
        bwrap_cmd.extend(["--dir", str(dir_name)])

    for env_var, value in env_vars.items():
        bwrap_cmd.extend(["--setenv", env_var, value])

    bwrap_cmd.extend(
        [
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--unshare-pid",
            "--die-with-parent",
            "--new-session",
            "--share-net",
        ]
    )

    bwrap_cmd.extend(["/bin/sh", "-c", shell_cmd])

    return bwrap_cmd
