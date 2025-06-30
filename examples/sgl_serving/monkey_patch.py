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

"""Patch sglang tokenizer endpoints."""

import logging

from fastapi import Request
from pydantic import BaseModel
from sglang.srt.openai_api.protocol import ChatCompletionMessageParam

logger = logging.getLogger(__name__)


class TokenizeRequest(BaseModel):
    text: str


class TokenizeMessageRequest(BaseModel):
    messages: list[ChatCompletionMessageParam]


class TokenizeResponse(BaseModel):
    text: str
    token_ids: list[int]


class DeTokenizeRequest(BaseModel):
    token_ids: list[int]


class DeTokenizeResponse(BaseModel):
    text: str
    token_ids: list[int]


def monkey_patch_tokenize_endpoint():
    from sglang.srt.entrypoints.http_server import app

    @app.post("/tokenize")
    async def tokenize(raw_request: Request) -> TokenizeResponse:
        from sglang.srt.entrypoints.http_server import _global_state

        request_json = await raw_request.json()
        request = TokenizeRequest(**request_json)
        token_ids = _global_state.tokenizer_manager.tokenizer.encode(request.text)

        return TokenizeResponse(token_ids=token_ids, text=request.text)

    @app.post("/detokenize")
    async def detokenize(raw_request: Request) -> DeTokenizeResponse:
        from sglang.srt.entrypoints.http_server import _global_state

        request_json = await raw_request.json()
        request = DeTokenizeRequest(**request_json)
        text = _global_state.tokenizer_manager.tokenizer.decode(request.token_ids)

        return DeTokenizeResponse(token_ids=request.token_ids, text=text)

    @app.post("/tokenize_with_template")
    async def tokenize_with_template(raw_request: Request) -> TokenizeResponse:
        from sglang.srt.entrypoints.http_server import _global_state

        request_json = await raw_request.json()
        request = TokenizeMessageRequest(**request_json)

        token_ids = _global_state.tokenizer_manager.tokenizer.apply_chat_template(
            request.messages, tokenize=True, add_generation_prompt=False
        )
        text = _global_state.tokenizer_manager.tokenizer.decode(token_ids)

        return TokenizeResponse(token_ids=token_ids, text=text)

    logger.info("Monkey patching tokenizer endpoints")
