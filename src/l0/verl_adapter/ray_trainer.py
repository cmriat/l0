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
"""Adapt PPO trainer."""

import uuid
import random
from pprint import pprint
from typing import Type, Optional
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from verl import DataProto
from torch.utils.data import Dataset, Sampler
from verl.utils.tracking import ValidationGenerationsLogger
from verl.single_controller.ray import RayWorkerGroup
from verl.single_controller.base import Worker
from verl.trainer.ppo.core_algos import agg_loss
from verl.utils.seqlen_balancing import log_seqlen_unbalance, get_seqlen_balanced_partitions
from verl.trainer.ppo.ray_trainer import (
    Role,
    RayPPOTrainer,
    ResourcePoolManager,
    _timer,
    compute_response_mask,
)
from verl.trainer.ppo.metric_utils import (
    reduce_metrics,
    compute_timing_metrics,
    compute_throughout_metrics,
    process_validation_metrics,
)

from l0.utils import pad_with_mask, unpad_with_mask
from l0.verl_adapter.advantage_calculation import (
    StepLevelAdvantageEstimator,
    compute_bi_level_advantage,
)

from .metric_utils import compute_data_metrics

WorkerType = Type[Worker]


class AgenticRayPPOTrainer(RayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
    ):
        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        # TODO: support reward model, like critic
        if self.use_rm:
            raise NotImplementedError("Reward model is not supported yet")

        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            raise NotImplementedError

        if self.config.algorithm.step_level_adv_estimator == StepLevelAdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.step_level_adv_estimator == StepLevelAdvantageEstimator.REINFORCE_PLUS_PLUS:
            self.use_critic = False
        else:
            raise NotImplementedError

        if self.config.algorithm.token_level_adv_estimator == "trivial":
            self.config.critic.use_token_level_value = False

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."
        )

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'"
                        + "is supported (the former is deprecated)."
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(
                config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic"
            )

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
            )

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            if config.actor_rollout_ref.traj_sampler.agent.get("max_steps", None) is not None:
                assert (
                    config.data.train_batch_size * config.actor_rollout_ref.traj_sampler.agent.max_steps
                    >= config.actor_rollout_ref.actor.ppo_mini_batch_size
                )
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert (
                    config.actor_rollout_ref.actor.ppo_mini_batch_size
                    % config.actor_rollout_ref.actor.ppo_micro_batch_size
                    == 0
                )
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            # assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (
            config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1
            or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1
        ):
            assert config.actor_rollout_ref.model.use_remove_padding, (
                "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
            )

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, (
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."
                )

        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated."
                + " Validation datasets are sent to inference engines as a whole batch,"
                + " which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _validate(self):
        data_source_lst = []
        metrics_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []

        for test_data in self.val_dataloader:
            print(f"test_batch_question length: {len(test_data['question'])}, example: {test_data['question'][:5]}")
            test_data["placeholder"] = torch.zeros(len(test_data["question"]), 1)
            test_batch = DataProto.from_single_dict(test_data)
            test_batch = pad_with_mask(test_batch, self.actor_rollout_wg._world_size)
            test_batch.non_tensor_batch["uuids"] = np.array(
                [str(uuid.uuid4()) + "_0" for _ in range(len(test_batch.batch))], dtype=object
            )
            id_to_batch = {
                test_batch.non_tensor_batch["uuids"][i]: test_batch[i : i + 1] for i in range(len(test_batch))
            }
            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            test_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }

            test_output_batch = self.actor_rollout_wg.generate_sequences(test_batch)
            if test_output_batch.batch is None:
                print(f"Collecting trajectories failed. Skipping step {self.global_steps}.")

            print("validation generation end")

            # Store generated outputs
            if test_output_batch.batch is None:
                print("No valid outputs generated. Skipping this batch.")
                continue
            output_ids = test_output_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            repeated_batches = []
            for i in range(len(test_output_batch)):
                repeated_batches.append(id_to_batch[test_output_batch.non_tensor_batch["uuids"][i]])
            # Concatenate all repeated batches
            test_batch = DataProto.concat(repeated_batches)
            test_batch.union(test_output_batch)
            test_batch = unpad_with_mask(test_batch, self.actor_rollout_wg._world_size)

            sample_inputs.extend(test_batch.non_tensor_batch["question"])

            # evaluate using reward_function

            reward_result = self.val_reward_fn(test_batch, return_dict=True)
            traj_score_info: dict = reward_result.get("traj_score_info", {})
            for key, val in traj_score_info.items():
                metrics_dict[key].extend(val)
            data_source_lst.append(
                traj_score_info.pop("data_source", "unknown_data_source" * len(traj_score_info["question"]))
            )

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, metrics_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "core_score"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens."""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=False
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """The training loop of PPO.

        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")  # noqa: T203
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for _ in range(self.config.trainer.total_epochs):
            # Initialize variables for filtering logic
            num_gen_batches = 0
            batch = None

            for batch_dict in self.train_dataloader:
                print(f"step:  {self.global_steps}, train_batch_question: {batch_dict['question'][:10]}")
                metrics = {}
                timing_raw = {}
                batch_dict["placeholder"] = torch.zeros(len(batch_dict["question"]), 1)
                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                new_batch.non_tensor_batch["uuids"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                )

                # repeat batch to align with rollout.n
                rollout_n = self.config.actor_rollout_ref.rollout.n
                gen_batch = new_batch.repeat(repeat_times=rollout_n, interleave=True)
                gen_uuids = []
                for i in range(len(new_batch.batch)):
                    base_uuid = new_batch.non_tensor_batch["uuids"][i]
                    for traj_id in range(rollout_n):
                        gen_uuids.append(f"{base_uuid}_{traj_id}")
                gen_batch.non_tensor_batch["uuids"] = np.array(gen_uuids, dtype=object)

                id_to_batch = {
                    gen_batch.non_tensor_batch["uuids"][i]: gen_batch[i : i + 1] for i in range(len(gen_batch))
                }
                is_last_step = self.global_steps >= self.total_training_steps
                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    if batch_output.batch is None:
                        print(f"Collecting trajectories failed. Skipping step {self.global_steps}.")
                        if is_last_step:
                            progress_bar.close()
                            return
                        progress_bar.update(1)
                        self.global_steps += 1
                        continue

                    repeated_batches = []
                    for i in range(len(batch_output)):
                        repeated_batches.append(id_to_batch[batch_output.non_tensor_batch["uuids"][i]])
                    new_batch = DataProto.concat(repeated_batches)
                    new_batch.union(batch_output)

                    # compute scores with rule-based rm
                    with _timer("reward", timing_raw):
                        reward_extra_infos_dict: dict[str, list]
                        reward_result = self.reward_fn(new_batch, return_dict=True)
                        token_level_reward_tensor = reward_result["token_level_reward_tensor"]
                        step_level_reward_tensor = reward_result["step_level_reward_tensor"]
                        reward_extra_infos_dict = reward_result["reward_extra_info"]
                        reward_metrics: dict = reward_result.get("metrics", {})
                        metrics.update(reward_metrics)
                        new_batch.batch["step_level_scores"] = step_level_reward_tensor  # for compatibility
                        new_batch.batch["token_level_scores"] = token_level_reward_tensor  # for compatibility
                        new_batch.batch["step_level_rewards"] = step_level_reward_tensor
                        new_batch.batch["token_level_rewards"] = token_level_reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:
                        num_gen_batches += 1

                        # Filter out questions where all metric_scores are 0 or all are 1
                        traj_score_info = reward_result.get("traj_score_info", {})
                        questions = traj_score_info.get("question", [])
                        metric_scores = traj_score_info.get(self.config.algorithm.filter_groups.metric, [])

                        # Create a mapping from question to metric_scores
                        question_to_metric_scores = {}
                        for question, metric_score in zip(questions, metric_scores, strict=False):
                            if question not in question_to_metric_scores:
                                question_to_metric_scores[question] = []
                            question_to_metric_scores[question].append(metric_score)

                        # Find questions to keep (not all 0 or all 1)
                        questions_to_keep = []
                        for question, metric_score in question_to_metric_scores.items():
                            # Keep if not all 0 and not all 1
                            if (
                                not (
                                    all(score == 0 for score in metric_score)
                                    or all(score == 1 for score in metric_score)
                                )
                                or random.random() > self.config.algorithm.filter_groups.drop_rate
                            ):
                                questions_to_keep.append(question)

                        # Create mask for filtering
                        keep_mask = np.array(
                            [q in questions_to_keep for q in new_batch.non_tensor_batch["question"]], dtype=bool
                        )

                        # Filter the batch using select_idxs
                        if keep_mask.any():
                            filtered_batch = new_batch.select_idxs(keep_mask)
                            batch = filtered_batch if batch is None else DataProto.concat([batch, filtered_batch])

                        # Check if we have enough samples after filtering
                        needed_bsz = self.config.data.train_batch_size * 2
                        if len(batch) < needed_bsz:
                            print(f"{len(batch)=} < {needed_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                if is_last_step:
                                    progress_bar.close()
                                    return
                                progress_bar.update(1)
                                self.global_steps += 1
                                continue  # Continue to next iteration of the for loop
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch = pad_with_mask(batch, self.actor_rollout_wg._world_size)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(
                            loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                        )
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    batch = unpad_with_mask(batch, self.actor_rollout_wg._world_size)

                    with _timer("adv", timing_raw):
                        batch = compute_bi_level_advantage(
                            batch,
                            config=self.config.algorithm,
                        )

                    batch = pad_with_mask(batch, self.actor_rollout_wg._world_size)

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                logger.log(data=metrics, step=self.global_steps)
                # Reset counters for next training step
                batch = None
                num_gen_batches = 0
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")  # noqa: T203
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
