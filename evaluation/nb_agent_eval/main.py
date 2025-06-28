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

"""Main entry point for evaluation."""

import os
import argparse

from evaluator import VALID_DATASETS, DirectEvaluator, NBAgentEvaluator
from omegaconf import OmegaConf

current_dir = os.path.dirname(__file__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run NBAgentEvaluator with specified configuration.")
    parser.add_argument("--datasets", type=str, nargs="+", choices=VALID_DATASETS, default=VALID_DATASETS)
    parser.add_argument("--config_path", type=str, default="config/sampler_config_train.yaml")
    return parser.parse_args()


def main(args):
    config = OmegaConf.load(args.config_path).traj_sampler

    if config.evaluator.mode == "nb_agent":
        evaluator = NBAgentEvaluator(config=config, dataset_names=args.datasets)
        evaluator.run(args.datasets)
    elif config.evaluator.mode == "direct":
        evaluator = DirectEvaluator(config=config, dataset_names=args.datasets)
        evaluator.run(args.datasets)
    else:
        raise ValueError(f"Unknown mode: {config.evaluator.mode}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
