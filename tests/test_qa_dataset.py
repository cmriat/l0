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
"""Test QA Dataset."""

import multiprocessing

from omegaconf import DictConfig
from transformers import AutoTokenizer

from l0.dataset.qa_dataset import QADataset

multiprocessing.set_start_method("spawn", force=True)

MODEL_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"
DATA_FILE = "/data/agent_datasets/qa_datasets/tmp_filtered/validation.parquet"


def test_qa_dataset_loading():
    """Test loading the QA dataset with a tokenizer and configuration."""
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_ID)
    dict_config = DictConfig({"max_prompt_length": 10000, "prompt_key": "question"})
    qa_dataset = QADataset(
        tokenizer=tokenizer,
        data_files=[DATA_FILE],
        config=dict_config,
    )
    for i in range(len(qa_dataset)):
        print(qa_dataset[i]["answer"])

    assert len(qa_dataset) > 0
    assert len(qa_dataset[0]) > 0
    assert qa_dataset[0]["answer"] is not None
    assert qa_dataset[0]["question"] is not None


if __name__ == "__main__":
    test_qa_dataset_loading()
