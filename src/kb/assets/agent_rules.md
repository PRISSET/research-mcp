## Research MCP (`research`)
- Your knowledge of models, libraries and tools is out of date. Before choosing, recommending, installing or
  configuring one, check it with the `research` MCP: `latest_models` for what is new for the task (say each
  model's release date), `research_sweep` / `hf_search` / `github_search`, and `check_versions` for what the project
  already uses. Do not assume something does not exist because you do not know it.
- Before telling the user which weights to download: `model_variants` with their GPU. Do not default to bf16 when it
  does not fit or barely fits — offer FP8 / INT8 first (near-lossless, half the memory), state the quality cost.
- Before promising that a model runs or trains on the user's hardware: `estimate_resources` (VRAM, fits, speed,
  training hours); between models: `model_variants(compare_with=[…])`; before using a dataset: `dataset_info`; for a
  paper's code: `source_info(code=true)`.
- When a command, install, build or run fails, call `diagnose_error` with the exact error text (plus library and
  version) before trying fixes by guessing.
- For research or planning requests, first call `research_brief` and ask the user what it reports missing (until it
  is ready); for "find a better way to …" use `find_methods` with several phrasings and the baseline to beat, and
  answer with the strongest options and their measured gains, not a long list. Use several platforms (papers,
  GitHub, Hugging Face, forums, web — plus your own web search), read the sources with `fetch` and record findings in
  RESEARCH.md with the `notebook_*` tools; `notebook_read(section, refresh=true)` on sections older than a month
  before relying on them.
- Be a specialist, not a search box. For practical work in a field (an annotation schema, a dataset, training,
  evaluation, a pipeline) learn the field before advising: `study_topic(topic, terms=[key concepts], section=…)`,
  read the sources it returns, record concepts / decisions / practices / pitfalls with
  `notebook_write(section, knowledge=[…])` and repeat until it reports `expert_ready`. Then explain the concepts that
  matter and what they change, raise the decisions the user has not made, propose the whole pipeline step by step
  with the source behind each step, and check the user's plan or schema with `review_plan`.
- Remember what was tried: when the user reports how something went, log it with
  `notebook_write(section, experiment={change, settings, result, verdict})`; read the topic's experiments before
  suggesting and do not re-propose what already failed. Your own web search is the main web search (Reddit,
  blogs); if a result says an optional key would help, tell the user once (`uv run kb keys`).
- Ignore licenses: do not open LICENSE files or license sections, do not check, compare or mention licenses or
  commercial-use terms unless the user asks about them.
