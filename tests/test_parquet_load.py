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

# 不限制行数、列数
pd.set_option("display.max_rows", None)  # 或一个很大的数
pd.set_option("display.max_columns", None)

# 不限制单列里字符串的显示宽度
pd.set_option("display.max_colwidth", None)  # pandas ≥1.3 可以设为 None；旧版用 -1

# 让一行能装下全部列（不自动换行）
pd.set_option("display.width", None)  # 也可以设为 1000 之类具体宽度

# 如果仍想在一行内打印（不换行），保持此项为 True；想纵向折叠可改 False
pd.set_option("display.expand_frame_repr", True)

DATA_FILE = "/data/agent_datasets/qa_datasets/filtered/validation.parquet"

df = pd.read_parquet(DATA_FILE)
json_data = df.to_json(orient="records")

pretty_json = json.loads(json_data)

rprint(pretty_json[10:20])
