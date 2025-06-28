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
"""This test is used to test the sglang server."""

import openai
import pytest


@pytest.mark.skip(reason="skip this test")
def test_sglang_server():
    client = openai.Client(base_url=f"http://127.0.0.1:60000/v1", api_key="None")

    response = client.chat.completions.create(
        model="Qwen/Qwen3-0.6B",
        messages=[
            {
                "role": "system",
                "content": "You need to trim the memory (context) of the agent to the most relevant information. The memory is a list of messages, each message is a dictionary contains a 'role' and a 'content', you have to extract and summarize the most relevant information and return it as a string. The agent will only see the system message and trimmed message returned by you, so, you need to make sure the trimmed message is concise and useful. Here are some categories of information that you can consider as most relevant: 1. Useful facts in previous observations 2. Meaningful actions that we've tired before 3. Lessons learned from previous attempts 4. Plans the agent has made before, and remained steps to be done The User may provide some extra requirements, you can consider them as well. But the piority of the extra requirements is lower than the instructions above.",
            },
            {
                "role": "system",
                "content": "You are an expert assistant aiming to solve task provided by user, you can interact with a jupyter notebook. You are required to conduct thorough research and deliberation to ensure the factuality, accuracy of the final answer. You need to conduct very in-depth exploration and reasoning, and detect all possible hidden or multi-hop relationships, so as to provide profound analysis and comprehensive answer. You have to be humble to your own knowledge, use the internet to check the facts. You must talk to the user in the user's language unless necessary. A notepad is provided, and it's content is the clearest memory of you and will always sticks to your cloest context. You could revise it via code and record most relevant and import information corresponding to your task including the 'todo list', 'important facts' and 'your draft'. Detailed manual cound be found in the tool description. Your memory have a token limitation, when the limitation is reached, all the memory except content in the notepad and user's task will be cleared, so you need to make full use of the notepad. You need to solve the task in steps, each step should consist of a '[THINK]:' field, a '[CODE]:' field and a '[OUTPUT]:' field: * Firstly, in the '[THINK]:' field, explain your reasoning and analysis towards solving the task at this step. * Then, in the '[CODE]:' field, write useful code in Python. The code block should begin with '```python' and end with '```<execute>'. Please make sure the code field ends with '<execute>', otherwise the code will not be executed. * Finally, code will be excuted as a cell in the notebook, and you will view the cell output in the '[OUTPUT]:' field, and will be available as input for the next step. You could expect display any legal format of jupyter notebook cell output such as images, strings, tables, etc. At the first step, you need to decompose the task and make a high-level plan to guide you and stick to it. You must claim the plan as a todo list and record it by calling `notepad.add_todo` method. You must update the status of corresponding todo item in 'notepad' based on you process after each step. You must update or revise your plan (todo list) based on your experience when necessary. You have to use `submit_final_answer` function to submit your final answer to user. Although user could see all your behavior, they may only focus on the final answer. So please make sure include all the information necessary while calling `submit_final_answer`. If the final answer is in words, please make it in a readable and pretty markdown format string.",
            },
            {"role": "user", "content": "# 任务 1:\nWhat's the first human land on the moon?"},
            {"role": "assistant", "content": "[THINK]: 为完成以上任务，先进行规划。"},
            {
                "role": "assistant",
                "content": "[CODE]:\n\n```python\nplanning(task='''What's the first human land on the moon?''')\n```",
            },
            {
                "role": "user",
                "content": "[OUTPUT]\n\nCONNECTIONERROR]\n('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))",
            },
        ],
        temperature=0,
        max_tokens=200,
    )

    print(f"Response: {response}")
    assert response is not None
    assert response.choices[0].message is not None
    assert response.choices[0].message.content is not None
    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].finish_reason is not None
    assert response.usage is not None
    assert response.usage.completion_tokens > 0


if __name__ == "__main__":
    test_sglang_server()
