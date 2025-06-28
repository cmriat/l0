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
"""Adapt FSDP sharding manager for SGLang in VERL."""

import os
import logging

from verl import DataProto
from sglang.srt.utils import broadcast_pyobj
from verl.workers.sharding_manager.fsdp_sglang import FSDPSGLangShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AgenticFSDPSGLangShardingManager(FSDPSGLangShardingManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def postprocess_data(self, data: DataProto) -> DataProto:
        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        device_mesh = self.inference_engine._device_mesh_cpu
        tp_rank = device_mesh.get_local_rank()
        tp_size = device_mesh.mesh.size()[0]
        data = broadcast_pyobj(
            data=[data],
            rank=device_mesh.get_rank(),
            dist_group=device_mesh.get_group(),
            src=device_mesh.mesh[0].item(),
            force_cpu_device=False,
        )[0]
        if tp_size > 1:
            logger.info(f"tp_size: {tp_size}, data_len: {len(data)}")
            try:
                data.meta_info["_verl_auto_padding"] = True
                local_prompts = data.chunk(chunks=tp_size)
                data = local_prompts[tp_rank]
            except Exception as e:
                logger.error(f"tp_size: {tp_size}, data_len: {len(data)}")
                logger.error(f"Error in postprocess_data: {e}")
        return data
