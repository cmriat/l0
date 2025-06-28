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
"""A worker that run NB Agent to sample trajectories."""

import hydra
from omegaconf import OmegaConf
from nbagent.agents import NBAgent
from nbagent.models import OpenAIServerModel
from hydra.core.config_store import ConfigStore

from .tool_specs import get_tool_specs
from .worker_config import NBAgentWorkerConfig

cs = ConfigStore.instance()
cs.store(name="config", node=NBAgentWorkerConfig)


@hydra.main(version_base=None, config_name="config")
def main(config: NBAgentWorkerConfig):
    config = OmegaConf.to_object(config)
    model = OpenAIServerModel(**config.openai_server_model.to_dict())
    nb_agent_config_dict = config.nb_agent.to_dict(shallow=True)
    tool_ability = nb_agent_config_dict.pop("tool_ability")
    nb_agent_config_dict.update({"tool_specs": get_tool_specs(tool_ability)})

    agent = NBAgent(
        model=model,
        **nb_agent_config_dict,
    )
    print(agent.sampling_params)
    config.task_run.task = config.task_run.task.replace("////", "//")
    config_dict = config.task_run.to_dict()
    config_dict.pop("file_path")
    agent.run(stream=False, **config_dict)


if __name__ == "__main__":
    main()
