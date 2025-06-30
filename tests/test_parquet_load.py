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

"""test parquet load."""

import json

import pandas as pd
from rich import print as rprint

# Do not limit the number of rows and columns
pd.set_option("display.max_rows", None)  # Or a very large number
pd.set_option("display.max_columns", None)

# Do not limit the display width of strings in a single column
pd.set_option("display.max_colwidth", None)  # pandas >=1.3 can be None; use -1 for older versions

# Allow all columns to fit on one line (no automatic wrapping)
pd.set_option("display.width", None)  # Can also be set to a specific width like 1000

# If you still want to print on one line (no wrapping), keep this as True; set to False for vertical folding
pd.set_option("display.expand_frame_repr", True)

DATA_FILE = "/data/agent_datasets/qa_datasets/filtered/validation.parquet"

df = pd.read_parquet(DATA_FILE)
json_data = df.to_json(orient="records")

pretty_json = json.loads(json_data)

rprint(pretty_json[10:20])
