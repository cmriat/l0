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

"""Patch sglang tokenizer manager."""

from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.tokenizer_manager import TokenizerManager


def monkey_patch_tokenizer_manager():
    def abort_request(self, rid: str):
        if rid == "":
            for rid in self.rid_to_state:
                req = AbortReq(rid)
                self.send_to_scheduler.send_pyobj(req)
        else:
            if rid not in self.rid_to_state:
                return
            req = AbortReq(rid)
            self.send_to_scheduler.send_pyobj(req)

    TokenizerManager.abort_request = abort_request
