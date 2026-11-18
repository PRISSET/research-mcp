## Research MCP (`research`)
- Your knowledge of models, libraries and tools is stale: before choosing, recommending or installing one, check it
  with the `research` MCP (`latest_models` with release dates, `hf_search`, `github_search`, `check_versions`).
  Weights for the user's GPU: `model_variants` (FP8 / INT8 before 4-bit, state the quality cost); will it fit / how
  fast: `estimate_resources`; training settings: `recipes(model=…)` with concrete values; a failed command:
  `diagnose_error` with the exact error first.
- Memory: `recall(query)` before re-asking the user or reading RESEARCH.md sections in full; `remember(topic,
  text)` for lasting facts (preference, decision and why, what worked or failed, where something is).
- The user's ComfyUI workflows and their measured runs: `workflows(goal)` — name the file and folder, lead with
  the run numbers and ratings, propose speed / quality compromises; save their verdict with `rate_run`.
- Research / planning: `research_brief` first and ask what it reports missing; practical work in a field:
  `study_topic` until `expert_ready`, then `review_plan`; log tries with `notebook_write(experiment=…)` and do not
  re-propose what failed.
- Reading-heavy jobs (many pages, issues, big files) go to `delegate(task=…, project=…)`: a Gemini subagent
  returns only facts with sources; check the sources before a claim decides something.
- Ignore licenses unless the user asks.
- Coding: record the user's requirements / constraints with `task_ledger`, reuse neighbouring helpers,
  `review_code` after each step, close every ledger item (done where / deferred why) and list them in the answer.
