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
"""Data parallel critic for agentic RL."""

import os
import logging

import torch
from verl.protocol import DataProto
from verl.trainer.ppo import core_algos
from verl.utils.debug import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import rearrange_micro_batches
from verl.utils.torch_functional import masked_mean
from verl.workers.critic.dp_critic import DataParallelPPOCritic

__all__ = ["DataParallelPPOCritic"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# TODO: could save a lot computation without token-level value prediction
class AgenticDataParallelCritic(DataParallelPPOCritic):
    """Data Parallel Critic for Agentic RL.

    This class extends the DataParallelPPOCritic to adapt it for Agentic RL.
    It overrides the `forward` method to handle the specific requirements of Agentic RL.
    """

    @GPUMemoryLogger(role="dp critic", logger=logger)
    def update_critic(self, data: DataProto):
        # make sure we are in training mode
        self.critic_module.train()
        metrics = {}

        select_keys = ["input_ids", "responses", "attention_mask", "position_ids", "token_level_values", "returns"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        ppo_mini_batch_size = len(batch) if self.config.do_not_split_ppo_mini_batch else self.config.ppo_mini_batch_size
        logger.info(f"Using ppo_mini_batch_size={ppo_mini_batch_size} for dp actor")

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(ppo_mini_batch_size)

        for _ in range(self.config.ppo_epochs):
            for _, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                    self.gradient_accumulation = ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu

                self.critic_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all devices
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(torch.cuda.current_device())  # critic device is cpu when using offload
                    responses = data["responses"]
                    attention_mask = data["attention_mask"]
                    values = data["token_level_values"]
                    returns = data["returns"]
                    response_length = responses.size(1)

                    response_mask = attention_mask[:, -response_length - 1 : -1]

                    if not self.config.use_token_level_value:
                        response_mask[:, 1:] = 0

                    vpreds = self._forward_micro_batch(data)

                    # assert not torch.any(torch.isnan(vpreds)).item()

                    vf_loss, vf_clipfrac = core_algos.compute_value_loss(
                        vpreds=vpreds,
                        values=values,
                        returns=returns,
                        response_mask=response_mask,
                        cliprange_value=self.config.cliprange_value,
                    )
                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = vf_loss * (len(data) / ppo_mini_batch_size)
                    else:
                        loss = vf_loss / self.gradient_accumulation

                    loss.backward()

                    data = {
                        "critic/vf_loss": vf_loss.detach().item(),
                        "critic/vf_clipfrac": vf_clipfrac.detach().item(),
                        "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
                    }

                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"critic/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.critic_optimizer.zero_grad()
        return metrics
