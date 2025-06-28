"""Launch the inference server."""

import os
import sys

from monkey_patch import monkey_patch_tokenize_endpoint

monkey_patch_tokenize_endpoint()

from sglang.srt.utils import kill_process_tree
from sglang.srt.server_args import prepare_server_args
from sglang.srt.entrypoints.http_server import launch_server

if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])

    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
