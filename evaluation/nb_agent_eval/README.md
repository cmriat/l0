# NB-Agent Evaluation

This repository contains a **light-weight yet scalable** evaluation harness for
NB-Agent (Notebook Agent)–style LLM systems.
It supports:

* **Multiple datasets** (Bamboogle, Musique, HotpotQA, SimpleQA, …)
* **Two evaluation modes**

  * `nb_agent` – use agent scafford for evaluation
  * `direct`   – a plain single-shot chat request
* **Local *or* Hugging Face (HF) dataset loading**
* **Remote execution** via an HTTP server (e.g. SG-Lang) with massive
  concurrency

---

## 1  Directory Layout

```text
evaluation/nb_agent_eval
├─ config/                 # YAML configs for the sampler & evaluator
│  ├─ eval_config_1.yaml
│  └─ ...
├─ eval_datasets/          # Dataset wrappers
│  ├─ __init__.py
│  ├─ _base.py
│  ├─ bamboogle.py
│  ├─ musique.py
│  └─ ...
├─ compute_metrices.py     # Metric implementations (EM, F1, etc.)
├─ evaluator.py            # Core evaluation engine
├─ sampler.py              # Trajectory sampler
├─ main.py                 # CLI entry point
└─ README.md               # ← **you are here**
```

---

## 2  Quick Start

```bash
# 1. init the environment
pixi install
pixi shell
```

### 2.1 Start Sglang serving
Since obtaining token usage information via sglang is not supported by default, we have patched the sglang server to add this functionality. To launch the modified sglang server, please follow the official documentation, but replace the default server script with `examples/sgl_serving/launch_server.py`.

### 2.3 Start task server
Similar to training, you need to start a task server to handle concurrent requests. You can refer to [training example document](../../README.md) for detailed instructions.

```bash
# 3. Edit config/sampler_config_train.yaml to point at your model / server

# 4. Back to evaluation directory and Run the example evaluation (4 datasets)
python main.py --datasets simpleqa bamboogle musique hotpotqa --config_path YOUR_CONFIG_PATH
```

`main.py` automatically picks up all listed datasets;
omit `--datasets` to evaluate **all** datasets registered in
`evaluator.VALID_DATASETS`.

---

## 3  Data Preparation

### Option A – Load from Hugging Face

*Set* `traj_sampler.evaluator.data_dir: "hf"` (default).
Datasets are fetched on-the-fly via `datasets.load_dataset(...)`.

### Option B – Load from Local Files

1. Convert each split to **newline-delimited JSON** (`*.jsonl`), one record per
   line with at least the following keys (just an example, exact schema should follow the DatasetSample define in each datasets loader under `eval_datasets/`):

   ```json
   {"id": "...", "question": "...", "golden_answers": "..."}
   ```

2. Place the files under an arbitrary folder, e.g.

   ```
   /data/bamboogle/test.jsonl
   /data/hotpotqa/validation.jsonl
   ...
   ```

3. In your config, set

   ```yaml
   evaluator:
     data_dir: "/data"         # absolute or relative path
   ```

The loader picks the local path whenever `data_dir` is **not** equal to
`"hf"`.

---

## 4  Configuration Reference

Below is a condensed explanation of every tunable field in
`config/sampler_config_train.yaml`.

| Key                                           | Type         | Default                         | Description                                                                |                                                            |
| --------------------------------------------- | ------------ | ------------------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------- |
| **traj\_sampler.model\_id**                   | *str*        | `"Qwen/Qwen2.5-7B-Instruct"`    | HF model name or local path passed to the remote server.                   |                                                            |
| **traj\_sampler.max\_concurrency**            | *int*        | `256`                           | Maximum *simultaneous* chat requests per process.                          |                                                            |
| **traj\_sampler.max\_retries**                | *int*        | `3`                             | How many times to retry a failed request before giving up.                 |                                                            |
| **traj\_sampler.task\_timeout**               | *int* (sec)  | `600`                           | Per-question wall-clock timeout (includes all retries).                    |                                                            |
| **traj\_sampler.agent\_type**                 | *str*        | `"nb_agent"`                    | Either `nb_agent` (multi-step) or `direct` (single call).                  |                                                            |
| **traj\_sampler.executor\_type**              | *str*        | `"remote"`                      | `remote` → call HTTP server; `local` → run in-process.                     |                                                            |
| **traj\_sampler.remote\_exec\_server\_url**   | *list\[str]* | —                               | One or more SG-Lang compatible endpoints for load-balancing.               |                                                            |
| **agent.uuid**                                | *str*        | `"abc"`                         | Unique experiment / run identifier (appears in logs & filenames).          |                                                            |
| **agent.openai\_server\_model.model\_id**     | *str*        | `"qwen2"`                       | Model alias recognised by your SG-Lang or OpenAI-compatible server.        |                                                            |
| **agent.openai\_server\_model.base\_url**     | *url*        | `"http://10.80.78.150:7999/v1"` | Base URL of the chat completion endpoint.                                  |                                                            |
| **agent.openai\_server\_model.server\_mode**  | *str*        | `"sglang"`                      | Adapter logic (`openai`, `sglang`, etc.).                                  |                                                            |
| **agent.nb\_agent.max\_tokens**               | *int*        | `32000`                         | Notebook memory token budget.                                              |                                                            |
| **agent.nb\_agent.max\_response\_length**     | *int*        | `3192`                          | Hard cap on **one** model response.                                        |                                                            |
| **agent.nb\_agent.traj\_save\_storage\_path** | *path*       | `"./trajectory"`                | Where to dump execution traces.                                            |                                                            |
| **agent.nb\_agent.agent\_workspace**          | *path*       | `"./agent_workspace"`           | Scratch directory for temporary files.                                     |                                                            |
| **agent.nb\_agent.agent\_mode**               | *str*        | `"llm"`                         | `llm` (pure LLM) or `hybrid` (LLM + tools).                                |                                                            |
| **agent.nb\_agent.trim\_memory\_mode**        | *str*        | `"hard"`                        | Memory pruning strategy (`none` / `soft` / `hard`).                        |                                                            |
| **agent.nb\_agent.tool\_ability**             | *str*        | `"qa"`                          | Tool subset enabled inside the notebook agent.                             |                                                            |
| **agent.nb\_agent.sampling\_params.top\_p**   | *float*      | `0.9`                           | Top-p nucleus sampling for creativity control.                             |                                                            |
| **task\_run.task**                            | *str*        | `"empty"`                         | Human-readable task name written to every trajectory. Will be reset at Runtime                      |                                                            |
| **task\_run.max\_steps**                      | *int*        | `10`                            | Maximum notebook steps per question (ignored in direct mode).              |                                                            |
| **task\_run.file\_path**                      | *path*       | `null`                          | left empty, will be automaticall set at runtime     |
| **log\_path**                                 | *path*       | `"./log"`                       | Folder for runtime logs.                                                   |                                                            |
| **evaluator.data\_dir**                       | *path*       | `"hf"(or local path)`           | `"hf"` = pull datasets online; otherwise read local files. |
| **evaluator.save\_dir**                       | *path*       | `"./results/<exp>"`             | All predictions + metrics are stored here.                                 |                                                            |
| **evaluator.mode**                            | *str*        | `"nb_agent"`                    | Must match `traj_sampler.agent_type` above.                                |                                                            |
| **evaluator.n\_samples**                      | *int*        | `3`                             | Number of **independent** runs per question to estimate stochastic models. |                                                            |

> **Tip** Any path may be absolute or relative to `evaluation/nb_agent_eval`.

---

## 5  Interpreting the Output

```text
results/
└─ qwen3-3b/
   ├─ bamboogle/
   │  ├─ result.jsonl   # each line = {"id": ..., "pred": ..., "label": ...}
   │  └─ logs/          # agent debug info
   |  └─ trajectory/    # agent trajectories
   ├─ musique/
   └─ ...
```

Run `python compute_metrices.py --result_path <JSONL>` to obtain
**Exact-Match (EM)** and **F1** scores. The `result_path` follow the previous example will be `results/qwen3-3b/`
You may aggregate scores across datasets with your own script or a spreadsheet.

---

## 6  Adding a New Dataset

1. Create `eval_datasets/<name>.py` inheriting from `BaseDataset` in
   `eval_datasets/_base.py`.
2. Append the class to the factory in `eval_datasets/__init__.py`.
3. Add `<name>` to `VALID_DATASETS` in `evaluator.py`.
4. Re-run `python main.py --datasets <name>`.

---

## 7  License

```
Copyright (c) 2022-2025
China Merchants Research Institute of Advanced Technology and Affiliates
Licensed under the Apache 2.0 License.
```
