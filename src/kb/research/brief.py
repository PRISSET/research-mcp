"""research_brief: before searching, make sure the task is specified well enough to find *strong* answers.

Vague requests produce vague research ("a list of candidates and some water"). The agent fills what it knows from
the conversation; every field that is still missing or vague comes back as a concrete question for the user. The
agent asks (several at once), calls again with the answers, and starts searching only when the brief is ready.
"""

from __future__ import annotations

import os
import re
from typing import Any

FIELDS: dict[str, dict[str, Any]] = {
    "goal": {
        "required": True,
        "ru": "Что именно нужно получить в итоге? (одно предложение: «улучшить X в Y», «выбрать Z для W»)",
        "en": "What exactly should the result be? (one sentence: 'improve X in Y', 'choose Z for W')",
    },
    "environment": {
        "required": True,
        "ru": "В чём это будет работать: модель / фреймворк / пайплайн и версии (например ComfyUI, diffusers, vLLM)?",
        "en": "Where will it run: model / framework / pipeline and versions (e.g. ComfyUI, diffusers, vLLM)?",
    },
    "baseline": {
        "required": True,
        "ru": "Что используется сейчас и что нужно превзойти (метод, настройки, текущий результат)?",
        "en": "What is used now and has to be beaten (method, settings, current result)?",
    },
    "success_metric": {
        "required": True,
        "ru": "По чему судить, что стало лучше: какая метрика или на что смотреть (визуал, звук, скорость, точность)?",
        "en": "How do we judge 'better': which metric or what to look at (visuals, audio, speed, accuracy)?",
    },
    "constraints": {
        "required": True,
        "ru": "Какие жертвы недопустимы: скорость (сколько % замедления ок), память / GPU, без дообучения, лицензия?",
        "en": "What must not get worse: speed (how much slowdown is acceptable), memory / GPU, no retraining, licence?",
    },
    "deliverable": {
        "required": False,
        "ru": "Что отдать в конце: короткий рейтинг решений, код / узел для пайплайна, план экспериментов?",
        "en": "What should be delivered: a short ranking, code / a node for the pipeline, an experiment plan?",
    },
    "already_tried": {
        "required": False,
        "ru": "Что уже пробовали и что не сработало (чтобы не предлагать это снова)?",
        "en": "What has already been tried and did not work (so it is not suggested again)?",
    },
}
VAGUE = re.compile(r"^\s*(\?+|-|n/?a|none|unknown|any|anything|whatever|tbd|не знаю|неизвестно|любой|любая|любое|"
                   r"что[- ]нибудь|как[- ]нибудь|нет)\s*\.?$", re.I)
# Follow-ups that matter only for some topics: (trigger in the text so far, field it must be mentioned in, question)
FOLLOW_UPS = [
    (r"(gpu|vram|cuda|rtx|a100|h100|weights?|model|модел|генерац|generat|inference|инференс|обучен|train|solver|солвер|"
     r"sampler|сэмплер|lora|comfyui|diffusers|vllm|llm|checkpoint|fp8|bf16|video|видео|image|изображ)",
     r"\b(\d+\s*(gb|гб)|rtx|gtx|a\d{2,3}|h\d{3}|l40|4090|3090|3060|5090|mac|m[1-4]\b|cpu only|только cpu)",
     {"ru": "Какая видеокарта и сколько видеопамяти (и RAM)?", "en": "Which GPU and how much VRAM (and RAM)?"}),
    (r"\b(speed|скорост|it/s|faster|быстр|latency|задержк|nfe|steps?|шаг)",
     r"(\d+\s*%|\bno slowdown|без замедлен|same (speed|nfe)|той же скорост|\d+\s*(s|с|сек|it/s|шаг|steps?)\b)",
     {"ru": "Сколько замедления допустимо в цифрах (0 %, до 10 %…) и при каком числе шагов сейчас?",
      "en": "How much slowdown is acceptable in numbers (0 %, up to 10 %...) and at how many steps now?"}),
    (r"\b(audio|звук|аудио|speech|речь|voice|голос)",
     r"(clarity|intelligib|разборчив|артефакт|artifact|lip[- ]?sync|синхрон|шум|noise|\bmos\b|\bwer\b|тембр|timbre)",
     {"ru": "Что именно важно в звуке: разборчивость речи, шумы и артефакты, синхронность с видео?",
      "en": "What matters in the audio: speech clarity, noise / artifacts, sync with the video?"}),
]


def _lang() -> str:
    return "ru" if os.environ.get("RESEARCH_LANG", "en").lower().startswith("ru") else "en"


def _filled(v: str | None) -> bool:
    return bool(v and len(v.strip()) >= 3 and not VAGUE.match(v))


def research_brief(topic: str, goal: str | None = None, environment: str | None = None, baseline: str | None = None,
                   success_metric: str | None = None, constraints: str | None = None, deliverable: str | None = None,
                   already_tried: str | None = None, user_declined: list[str] | None = None) -> dict[str, Any]:
    lang = _lang()
    values = {"goal": goal, "environment": environment, "baseline": baseline, "success_metric": success_metric,
              "constraints": constraints, "deliverable": deliverable, "already_tried": already_tried}
    declined = set(user_declined or [])
    missing, questions = [], []
    for name, spec in FIELDS.items():
        if spec["required"] and not _filled(values[name]) and name not in declined:
            missing.append(name)
            questions.append({"field": name, "question": spec[lang]})
    text = " ".join([topic] + [v for v in values.values() if v]).lower()
    for i, (trigger, answered, q) in enumerate(FOLLOW_UPS):
        key = f"detail_{i}"
        if re.search(trigger, text) and not re.search(answered, text) and key not in declined:
            missing.append(key)
            questions.append({"field": key, "question": q[lang]})
    optional = [{"field": n, "question": FIELDS[n][lang]} for n in ("deliverable", "already_tried")
                if not _filled(values[n]) and n not in declined]
    ready = not missing
    brief = "\n".join(f"- **{n}:** {v.strip()}" for n, v in values.items() if _filled(v))
    return {
        "ready": ready,
        "missing": missing,
        "questions": questions,
        "optional_questions": optional if not ready else [],
        "brief": brief,
        "next": (
            "Ready: notebook_write(title=…, question=<goal + this brief>), then search. Put success_metric and "
            "constraints into every judgement: drop candidates that break a constraint."
            if ready else
            "Ask the user these questions now (all at once; with AskUserQuestion if you have it), offering likely "
            "options where you can. Do not guess the answers and do not start searching. Then call research_brief "
            "again with the answers; if the user does not know or does not care, list the field in user_declined."
        ),
    }
