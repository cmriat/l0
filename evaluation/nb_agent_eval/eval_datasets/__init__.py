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
"""Evaluation datasets package."""

from .nq import NQDataset
from .gaia import GAIADataset
from ._base import BaseDataset
from .musique import MusiqueDataset
from .hotpotqa import HotpotQADataset
from .simpleqa import SimpleQADataset
from .triviaqa import TriviaQADataset
from .bamboogle import BamboogleDataset
from .wikimultihop import WikiMultihopQADataset

__all__ = [
    "NQDataset",
    "SimpleQADataset",
    "MusiqueDataset",
    "BamboogleDataset",
    "GAIADataset",
    "TriviaQADataset",
    "WikiMultihopQADataset",
    "HotpotQADataset",
    "SimpleQADataset",
    "BaseDataset",
]
