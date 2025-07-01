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
"""Tool specs of NB agent."""

import os
import logging
from typing import Any

TOOL_FACTORY_MAP: dict[str, set[str]] = {"qa": {"web_search_tool_factory", "jina_reader_tool_factory", "visual_qa_tool_factory"}, "math": {}}

TOOL_SPECS_MAP: dict[str, dict[str, Any]] = {
    "file_inspector_tool_factory": {
        "model_id": os.getenv("OPENAI_MODEL_ID"),
        "base_url": os.getenv("OPENAI_API_BASE"),
        "api_key": os.getenv("OPENAI_API_KEY"),
    },
    "exa_search_tool_factory": {
        "exa_api_key": os.getenv("EXA_API_KEY"),
    },
    "firecrawl_search_tool_factory": {
        "firecrawl_base_url": os.getenv("FIRECRAWL_BASE_URL"),
    },
    "web_search_tool_factory": {"serper_api_key": os.getenv("SERPER_API_KEY"), "max_text_limitation": 1000},
    "jina_reader_tool_factory": {"api_key": os.getenv("JINA_API_KEY"), "token_budget": 100000},
    "visual_qa_tool_factory": {"model_id": os.getenv("VLM_MODEL_ID"), "base_url": os.getenv("VLM_API_BASE"), "api_key": os.getenv("VLM_API_KEY")}
}


def get_tool_specs(tool_ability: str) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    tool_factories = TOOL_FACTORY_MAP.get(tool_ability, {})
    for tool_factory in tool_factories:
        if tool_factory in TOOL_SPECS_MAP:
            specs[tool_factory] = TOOL_SPECS_MAP[tool_factory]
        else:
            logging.warning(f"No specs found for tool factory: {tool_factory}")
    return specs
