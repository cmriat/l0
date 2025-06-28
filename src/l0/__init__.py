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
"""L0: Learn to for training agents in RL."""

import dotenv
from omegaconf import OmegaConf

dotenv.load_dotenv()

# Define and register a custom resolver for addition
OmegaConf.register_new_resolver("add", lambda *numbers: sum(numbers))

# Import sglang_adapter to ensure monkey patching is applied
from .sglang_adapter import monkey_patch_sglang

monkey_patch_sglang()
