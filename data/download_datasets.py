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
"""Download and process datasets."""

import os
import argparse

from tqdm import tqdm
from upath import UPath
from data_processing.downloader import DOWNLOADER_DICT, DatasetDownloader


def setup_proxy(http_proxy: str) -> None:
    """Set up proxy for downloading datasets.

    Args:
        http_proxy: HTTP proxy URL
    """
    os.environ["https_proxy"] = http_proxy
    os.environ["http_proxy"] = http_proxy
    os.environ["HTTP_PROXY"] = http_proxy
    os.environ["HTTPS_PROXY"] = http_proxy
    print(f"Proxy set to {http_proxy}")


def download_datasets(
    cache_dir: UPath,
    save_dir: UPath,
    proxy: str | None,
    downloader_dict: dict[str, DatasetDownloader],
    num_proc: int = 8,
) -> None:
    """Process all datasets.

    Args:
        cache_dir: Directory to cache raw datasets
        save_dir: Directory to save processed datasets
        proxy: HTTP proxy URL
        num_proc: Number of processes to use for downloading
    """
    if proxy is not None:
        setup_proxy(proxy)

    for dataset_name, downloader_cls in tqdm(downloader_dict.items(), desc="Processing datasets"):
        downloader = downloader_cls(cache_dir, save_dir, num_proc=num_proc)
        print(f"\n=== Processing {dataset_name} ===")
        downloader.download()


def get_parser():
    parser = argparse.ArgumentParser(description="Download and process datasets")
    parser.add_argument("--cache_dir", required=True, help="Directory to cache raw datasets")
    parser.add_argument("--save_dir", required=True, help="Directory to save modified datasets")

    dataset_choices = [
        "all",
        "simple_qa",
        "natural_questions",
        "trivia_qa",
        "hotpot_qa",
        "wiki_multihop",
        "bamboogle",
        "musique",
        "pop_qa",
    ]
    parser.add_argument(
        "--dataset",
        choices=dataset_choices,
        nargs="+",  # 允许一个或多个参数
        default=["all"],  # 默认值现在是一个列表
        help="Dataset(s) to process (e.g., simple_qa natural_questions or all) (default: all)",
    )
    parser.add_argument("--http_proxy", default=None, help="HTTP proxy to use")
    parser.add_argument(
        "--num_proc",
        type=int,
        default=8,
        help="Number of processes to use for downloading (default: 8)",
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    args.cache_dir = UPath(args.cache_dir)
    args.save_dir = UPath(args.save_dir)
    # Process selected dataset(s)
    if "all" in args.dataset:
        downloader_dict = DOWNLOADER_DICT
    else:
        downloader_dict = {name: DOWNLOADER_DICT[name] for name in args.dataset if name in DOWNLOADER_DICT}
        if not downloader_dict:
            print(f"No valid datasets selected from: {args.dataset}. Available: {list(DOWNLOADER_DICT.keys())}")
            exit(1)

    download_datasets(
        args.cache_dir, args.save_dir, args.http_proxy, downloader_dict=downloader_dict, num_proc=args.num_proc
    )
