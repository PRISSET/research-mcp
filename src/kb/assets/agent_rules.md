## Research MCP (`research`)
- Your knowledge of models, libraries and tools is out of date. Before choosing, recommending, installing or
  configuring one, check it with the `research` MCP: `latest_models` for what is new for the task (say each
  model's release date), `research_sweep` / `hf_search` / `github_search`, and `check_versions` for what the project
  already uses. Do not assume something does not exist because you do not know it.
- Before telling the user which weights to download: `model_variants` with their GPU. Do not default to bf16 when it
  does not fit or barely fits — offer FP8 / INT8 first (near-lossless, half the memory), state the quality cost.
- Before promising that a model runs or trains on the user's hardware: `estimate_resources` (VRAM, fits, speed,
  training hours); between models: `compare_models`; before using a dataset: `dataset_info`; for a paper's code:
  `paper_code`.
- When a command, install, build or run fails, call `diagnose_error` with the exact error text (plus library and
  version) before trying fixes by guessing.
- For research or planning requests, first call `research_brief` and ask the user what it reports missing (until it
  is ready); for "find a better way to …" use `find_methods` with several phrasings and the baseline to beat, and
  answer with the strongest options and their measured gains, not a long list. Use several platforms (papers,
  GitHub, Hugging Face, forums, web — plus your own web search), read the sources (`fetch_url`, `fetch_fulltext`) and record findings in RESEARCH.md with the
  `notebook_*` tools; `notebook_refresh` sections older than a month before relying on them.
