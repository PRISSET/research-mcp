"""Tests never see the user's own keys file or keys exported in the shell."""

import os

os.environ["RESEARCH_KEYS"] = os.devnull
for _k in ("SEARXNG_URL", "TAVILY_API_KEY", "BRAVE_API_KEY", "GITHUB_TOKEN", "HF_TOKEN", "S2_API_KEY",
           "OPENALEX_API_KEY", "CIVITAI_API_KEY", "RESEARCH_NSFW"):
    os.environ.pop(_k, None)
