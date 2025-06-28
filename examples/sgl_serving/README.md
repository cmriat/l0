# Launching patched SGLang server

* Since `NB-Agent` needs to use the tokenizer of the model, we patch SGLang to provide extra endpoints.
* Simply `cd` to this directory replace `python3 -m sglang.launch_server {other args}` with `python launch_server.py {other args}` to launch the patched SGLang server.