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

"""Dataset downloaders."""

from .base_downloader import DatasetDownloader
from .popqa_downloader import PopQADownloader
from .musique_downloader import MusiqueDownloader
from .hotpotqa_downloader import HotpotQADownloader
from .simpleqa_downloader import SimpleQADownloader
from .triviaqa_downloader import TriviaQADownloader
from .bamboogle_downloader import BamboogleDownloader
from .wikimultihopqa_downloader import WikiMultiHopQADownloader
from .natural_questions_downloader import NaturalQuestionsDownloader

__all__ = [
    "DatasetDownloader",
    "PopQADownloader",
    "MusiqueDownloader",
    "HotpotQADownloader",
    "SimpleQADownloader",
    "TriviaQADownloader",
    "BamboogleDownloader",
    "WikiMultiHopQADownloader",
    "NaturalQuestionsDownloader",
]

DOWNLOADER_DICT = {
    "pop_qa": PopQADownloader,
    "musique": MusiqueDownloader,
    "hotpot_qa": HotpotQADownloader,
    "simple_qa": SimpleQADownloader,
    "trivia_qa": TriviaQADownloader,
    "bamboogle": BamboogleDownloader,
    "wiki_multihop": WikiMultiHopQADownloader,
    "natural_questions": NaturalQuestionsDownloader,
}
