"""study_topic / review_plan: learn a field like a specialist before advising, then hold plans against it.

find_methods answers "what beats X"; it does not teach the craft. A specialist in, say, video annotation knows the
concepts (atomic actions, granularity), the decisions every project has to make (label vocabulary, temporal
boundaries, agreement checks), the practices of the reference datasets and the usual failures. Here:
  1. the topic is split into facets the way a specialist would study it (perspective-guided, as in STORM):
     foundations, reference datasets / systems, each concept the agent names, and the decisions on the checklist
     of the task kind (annotation, training, evaluation, dataset, inference);
  2. every facet is searched (papers ranked by meaning and citations — surveys and canonical dataset papers are
     wanted here, unlike in find_methods; web guides; repos);
  3. the best sources are read in full right away and the passages that answer each facet are returned;
  4. what the agent learns goes into the topic's "specialist knowledge" in RESEARCH.md with its sources, and
     the call reports whether that knowledge is deep enough to advise (`expert_ready`), like research_brief does
     for the question itself.
review_plan then checks a plan, schema or pipeline against the checklist and the learned pitfalls / practices.
"""

from __future__ import annotations

import logging
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import date
from typing import Any

from kb.schemas import Paper
from kb.store import Database

log = logging.getLogger(__name__)

SEARCH_DEADLINE = 45.0
READ_DEADLINE = 90.0


@dataclass(frozen=True)
class Check:
    id: str
    query: str        # English search phrase, {t} is the topic; empty: settled with the user, nothing to read
    ru: str           # the decision a specialist settles, and why it matters
    en: str
    mentioned: str    # regex: the plan already says something about it
    when: str = ""    # regex on the topic / plan: only then is the check relevant


PROFILES: dict[str, dict[str, Any]] = {
    "annotation": {
        "match": r"annot|label|разметк|размеч|caption|подпис|schema|схем|taxonom|ontolog|таксоном|tagging|теги",
        "checks": [
            Check("downstream", "annotation design for training {t} models",
                  "Для чего разметка: какую модель она учит или оценивает и в каком виде метки нужны на выходе (формат "
                  "подписи должен совпадать с промптами на инференсе)? От этого зависят все остальные решения.",
                  "What are the labels for: which model they train or evaluate and in what form they are consumed "
                  "(captions must match the prompts used at inference)? Every other decision follows from it.",
                  r"train|обуч|lora|inference|инференс|prompt|промпт|downstream"),
            Check("granularity", "{t} annotation granularity hierarchy activity action atomic",
                  "Единица и гранулярность: что получает одну метку (клип, отрезок, кадр, событие) и на каком уровне "
                  "(деятельность → действие → атомарное движение). Уровень выбирают под задачу модели и проверяют, что "
                  "люди различают его стабильно.",
                  "Unit and granularity: what gets one label (clip, segment, frame, event) and at which level "
                  "(activity → action → atomic action). Pick the level the model needs and check people separate it "
                  "consistently.",
                  r"granular|гранул|atomic|атомар|segment|сегмент|отрез|primitive|примитив|уровн|level"),
            Check("vocabulary", "{t} label taxonomy design closed vocabulary",
                  "Словарь меток: закрытый список; классы одного уровня взаимоисключают друг друга и вместе покрывают "
                  "все случаи; есть «другое» / «не видно» / «неясно». Иначе разметчики придумывают свои ответы, а "
                  "модель учит шум.",
                  "Label vocabulary: a closed list; classes at one level are mutually exclusive and together cover "
                  "every case; there is 'other' / 'not visible' / 'unclear'. Otherwise annotators improvise and the "
                  "model learns noise.",
                  r"other|unclear|unknown|not visible|n/a|друго|не видно|неясн|нет в кадре"),
            Check("factorize", "compositional action annotation verb noun attributes",
                  "Классы отделены от атрибутов: действие, объект, исполнитель, манера и скорость — отдельные поля, а "
                  "не склеенные метки. Так словарь не разрастается, а редкие сочетания всё равно описываются.",
                  "Classes separate from attributes: action, object, actor, manner and speed as separate fields, not "
                  "glued labels. The vocabulary stays small and rare combinations are still expressible.",
                  r"attribut|атрибут|field|поле|поля|slot|слот|verb|глагол"),
            Check("boundaries", "temporal boundary ambiguity action annotation",
                  "Правила границ во времени: где движение начинается и кончается, минимальная длительность, "
                  "перекрытия и одновременные действия (несколько меток сразу). Границы — главный источник расхождений "
                  "между разметчиками.",
                  "Temporal boundary rules: where an action starts and ends, minimum duration, overlaps and "
                  "simultaneous actions (multi-label). Boundaries are the main source of disagreement between "
                  "annotators.",
                  r"start|begin|end\b|boundar|overlap|начал|конец|окончан|границ|перекр|таймкод|timestamp",
                  r"video|видео|clip|клип|temporal|времен|action|движен|действ|audio|звук"),
            Check("guidelines", "{t} annotation guidelines edge cases examples",
                  "Инструкция разметчика: определение каждой метки, примеры «да» и «нет», пограничные случаи и как их "
                  "решать. Версия инструкции хранится вместе с метками.",
                  "Annotator guidelines: a definition of every label, positive and negative examples, edge cases and "
                  "how to resolve them; the guideline version is stored with the labels.",
                  r"guideline|instruction|инструкц|пример|example|edge case|погранич|определен"),
            Check("agreement", "{t} inter-annotator agreement",
                  "Пилот и согласованность: несколько человек независимо размечают одну выборку, считают κ Коэна / "
                  "α Криппендорфа (для отрезков — temporal IoU); метки с низким согласием переопределяют до разметки "
                  "всего набора.",
                  "Pilot and agreement: several people label the same sample independently, compute Cohen's κ / "
                  "Krippendorff's α (temporal IoU for segments); redefine low-agreement labels before labelling "
                  "everything.",
                  r"agreement|согласован|kappa|каппа|κ|alpha|альфа|iou|pilot|пилот"),
            Check("quality", "{t} annotation quality control gold standard",
                  "Контроль качества: эталонный набор (gold), скрытые контрольные примеры, выборочная перепроверка и "
                  "правило разрешения споров.",
                  "Quality control: a gold set, hidden control items, spot re-checks and an adjudication rule for "
                  "disagreements.",
                  r"gold|эталон|honeypot|контрольн|spot.?check|перепровер|adjudicat|арбитр|review|ревью"),
            Check("prelabel", "pre-annotation effects on human annotators",
                  "Предразметка моделью: люди склонны соглашаться с подсказкой. Точность модели меряют на эталоне по "
                  "каждому полю и подсказывают только там, где она точна.",
                  "Model pre-labelling: people tend to accept the suggestion. Measure the model per field on the gold "
                  "set and pre-fill only the fields where it is accurate.",
                  r"pre-?label|pre-?annot|предразмет|подсказк|suggest|vlm|модел"),
            Check("balance", "{t} class imbalance long tail dataset",
                  "Распределение меток: примеров на класс, редкие классы, перекосы. Для генеративных LoRA ещё и "
                  "разнообразие остальных признаков, чтобы концепт не склеился с одним фоном или позой.",
                  "Label distribution: examples per class, rare classes, skew. For generative LoRAs also variety in "
                  "everything else, so the concept is not tied to one background or pose.",
                  r"balanc|баланс|distribut|распредел|rare|редк|long.?tail|перекос|разнообраз|divers"),
            Check("leakage", "dataset split leakage near-duplicate clips same source video",
                  "Разбиение и утечки: клипы одного исходного видео не попадают одновременно в обучение и проверку; "
                  "дубли и почти-дубли убраны.",
                  "Splits and leakage: clips of one source video never land in both train and validation; duplicates "
                  "and near-duplicates removed.",
                  r"split|leak|утечк|holdout|отложен|валидац|validation|дубл|duplicat"),
            Check("versioning", "",
                  "Версии схемы: номер версии у каждой метки и правило переноса старых меток при изменении словаря.",
                  "Schema versions: a version on every label and a rule for migrating old labels when the vocabulary "
                  "changes.",
                  r"version|верси|v\d|migrat|миграц"),
            Check("cost", "{t} annotation cost time",
                  "Цена и удобство: время на единицу, сколько полей за проход (несколько проходов по одному аспекту "
                  "обычно быстрее и точнее), горячие клавиши, порядок полей.",
                  "Cost and ergonomics: time per item, how many fields per pass (several single-aspect passes are "
                  "usually faster and more accurate), hotkeys, field order.",
                  r"time per|время на|pass|проход|hotkey|горяч|interface|интерфейс|ui\b"),
            Check("captions", "caption structure for text-to-video text-to-image training data",
                  "Подписи для генеративной модели: фиксированный порядок частей, единый словарь, триггер-слово, длина в "
                  "пределах текстового энкодера; что описывать (то, что меняется) и что нет (то, что должна выучить "
                  "LoRA).",
                  "Captions for a generative model: fixed part order, one vocabulary, trigger word, length within the "
                  "text encoder; describe what varies, leave out what the LoRA should learn.",
                  r"trigger|триггер|order|порядок|token|токен|length|длин|template|шаблон",
                  r"caption|подпис|t2v|t2i|lora|text-to|генерац|generat"),
        ],
    },
    "training": {
        "match": r"train|fine-?tun|lora|обуч|дообуч|dreambooth|distill|дистилл",
        "checks": [
            Check("data", "{t} training data quality quantity",
                  "Данные: сколько примеров, их качество и разнообразие, дубликаты, разрешение; что из этого сильнее "
                  "всего влияет на результат по источникам.",
                  "Data: how many examples, their quality and variety, duplicates, resolution; which of these matters "
                  "most according to the sources.",
                  r"dataset|датасет|images|изображ|clips|клип|examples|пример"),
            Check("base", "{t} base model choice",
                  "Базовая модель и её точность / квант: та же, что будет на инференсе?",
                  "Base model and its precision / quantisation: the same one used at inference?",
                  r"base|базов|checkpoint|чекпоинт|fp8|bf16|int8"),
            Check("hparams", "{t} hyperparameters learning rate rank steps",
                  "Гиперпараметры: LR, rank / alpha, шаги или эпохи, batch, оптимизатор — от каких проверенных "
                  "значений отталкиваемся и почему.",
                  "Hyperparameters: LR, rank / alpha, steps or epochs, batch, optimiser — which proven values we start "
                  "from and why.",
                  r"lr\b|learning rate|rank|alpha|steps|шаг|epoch|эпох|batch|optimi"),
            Check("validation", "{t} overfitting validation during training",
                  "Проверка по ходу: фиксированные промпты и сиды каждые N шагов, отложенные примеры и признак, когда "
                  "остановиться (копирует датасет, теряет гибкость).",
                  "Checks during training: fixed prompts and seeds every N steps, held-out examples and a stopping "
                  "signal (copies the dataset, loses flexibility).",
                  r"validat|валидац|seed|сид|sample every|overfit|переобуч|checkpoint every"),
            Check("ablation", "{t} ablation study",
                  "Эксперименты: одно изменение за раз, конфиги и результаты записаны, есть с чем сравнить.",
                  "Experiments: one change at a time, configs and results recorded, a baseline to compare with.",
                  r"ablat|абляц|one change|одно изменен|baseline|бейзлайн|сравн"),
        ],
    },
    "evaluation": {
        "match": r"\beval|benchmark|бенчмарк|оцен|метрик|metric",
        "checks": [
            Check("metric", "{t} evaluation metric correlation with human judgment",
                  "Метрика: что именно она меряет и насколько совпадает с оценкой людей на этой задаче.",
                  "Metric: what exactly it measures and how well it agrees with human judgement on this task.",
                  r"metric|метрик|correlat|коррел|human|человек"),
            Check("human", "{t} human evaluation protocol",
                  "Оценка людьми: слепое сравнение, сколько оценщиков, критерии, согласие между ними.",
                  "Human evaluation: blind comparison, how many raters, criteria, agreement between them.",
                  r"blind|слеп|rater|оценщик|side.?by.?side|pairwise|попарн"),
            Check("variance", "{t} evaluation variance seeds statistical significance",
                  "Разброс: несколько сидов, доверительные интервалы; разница больше шума?",
                  "Variance: several seeds, confidence intervals; is the difference larger than the noise?",
                  r"seed|сид|variance|разброс|confidence|интервал|significan|значим"),
            Check("contamination", "{t} benchmark contamination held-out",
                  "Чистота проверки: тестовые примеры не видела ни модель, ни настройка гиперпараметров.",
                  "Clean test: neither the model nor the hyperparameter tuning saw the test items.",
                  r"contamin|загрязн|held.?out|отложен|test set|тестов"),
        ],
    },
    "dataset": {
        "match": r"dataset|датасет|curat|курац|dedup|дедуп|corpus|корпус|filtering|фильтрац",
        "checks": [
            Check("sourcing", "{t} dataset collection licensing",
                  "Источники и права: откуда данные, лицензия, согласие, что нельзя публиковать.",
                  "Sources and rights: where the data comes from, licence, consent, what cannot be published.",
                  r"licen|лиценз|source|источник|consent|соглас"),
            Check("filtering", "{t} data filtering quality deduplication",
                  "Фильтрация: дубли и почти-дубли, качество (резкость, водяные знаки, обрезка), автоматические фильтры "
                  "и их пороги.",
                  "Filtering: duplicates and near-duplicates, quality (sharpness, watermarks, crops), automatic filters "
                  "and their thresholds.",
                  r"dedup|дубл|filter|фильтр|quality|качеств|watermark|водян"),
            Check("documentation", "datasheets for datasets documentation",
                  "Описание набора: состав, как собран и размечен, известные перекосы — чтобы через месяц было понятно, "
                  "что в нём.",
                  "Dataset documentation: composition, how it was collected and labelled, known biases — so it is "
                  "still understood a month later.",
                  r"datasheet|readme|документ|описан"),
        ],
    },
    "inference": {
        "match": r"inference|инференс|deploy|serving|latency|задержк|throughput|ускор|speed.?up|pipeline|пайплайн|"
                 r"comfyui|vllm",
        "checks": [
            Check("reference", "{t} quality degradation quantization acceleration comparison",
                  "Эталон качества: каждое ускорение или квант сравнивается с исходной конфигурацией на одних промптах "
                  "и сидах.",
                  "Quality reference: every speed-up or quantisation is compared with the original setup on the same "
                  "prompts and seeds.",
                  r"reference|эталон|same seed|те же сид|compare|сравн|baseline"),
            Check("profile", "{t} profiling bottleneck",
                  "Узкое место измерено (профайлер, время по стадиям), а не угадано.",
                  "The bottleneck is measured (profiler, time per stage), not guessed.",
                  r"profil|профил|bottleneck|узк|по стадиям|per stage"),
            Check("failures", "{t} failure handling retries monitoring",
                  "Сбои: что делать при OOM, таймауте, битом входе; логирование, повтор, мониторинг.",
                  "Failures: what happens on OOM, timeout, bad input; logging, retries, monitoring.",
                  r"oom|retry|повтор|timeout|таймаут|monitor|монитор|лог"),
        ],
    },
    "generic": {
        "match": r"",
        "checks": [
            Check("prior_art", "{t} state of the art approaches",
                  "Как это решают лучшие: эталонные работы и системы, чем отличаются их подходы.",
                  "How the best solve it: reference works and systems, how their approaches differ.",
                  r"prior|existing|аналог|существующ|state of the art|sota"),
            Check("failure_modes", "{t} failure modes limitations",
                  "Типичные провалы этого подхода и как их заметить заранее.",
                  "Typical failures of this approach and how to notice them early.",
                  r"fail|провал|limitation|огранич|risk|риск"),
            Check("decisive", "{t} evaluation experiment",
                  "Какой небольшой эксперимент быстро покажет, что выбранный путь работает.",
                  "Which small experiment will quickly show that the chosen path works.",
                  r"experiment|экспер|pilot|пилот|test|тест|провер"),
        ],
    },
}

# What a specialist knowledge base must hold before advising: per kind of item, plus sources actually read.
GATES = {
    "quick": {"concept": 2, "decision": 2, "practice": 3, "pitfall": 2, "read": 5},
    "normal": {"concept": 3, "decision": 3, "practice": 5, "pitfall": 3, "read": 8},
    "deep": {"concept": 5, "decision": 4, "practice": 8, "pitfall": 5, "read": 15},
}
SURVEY = re.compile(r"\b(survey|review|overview|tutorial|taxonomy|primer)\b", re.I)


def _lang() -> str:
    return "ru" if os.environ.get("RESEARCH_LANG", "en").lower().startswith("ru") else "en"


def profiles_for(text: str, kinds: list[str] | None = None) -> list[str]:
    if kinds:
        chosen = [k for k in kinds if k in PROFILES]
        if chosen:
            return chosen
    low = text.lower()
    found = [k for k, p in PROFILES.items() if p["match"] and re.search(p["match"], low)]
    return found[:3] or ["generic"]


def checks_for(profiles: list[str], text: str) -> list[Check]:
    low = text.lower()
    out: dict[str, Check] = {}
    for name in profiles:
        for c in PROFILES[name]["checks"]:
            if not c.when or re.search(c.when, low):
                out.setdefault(c.id, c)
    return list(out.values())


STOP = {"the", "and", "for", "with", "from", "how", "what", "which", "its", "are", "into", "that", "this", "when",
        "why", "use", "using", "based", "toward", "towards", "via", "best", "common", "recent", "new", "study"}


def _words(text: str) -> set[str]:
    return {w.rstrip("s") if len(w) > 4 else w for w in re.findall(r"[a-z][a-z0-9]{2,}", (text or "").lower())
            if w not in STOP}


def _cover(query_words: set[str], text: str) -> float:
    return len(query_words & _words(text)) / max(1, len(query_words))


def facets(topic: str, terms: list[str] | None, checks: list[Check]) -> list[dict[str, Any]]:
    """What to study, in the order a specialist would: each facet = searches + the question its passages answer.
    Paper searches are (query, mode): "rel" relevance, "cited" most cited (canonical works), "recent"."""
    t = topic.strip()
    head = t.split()[0]
    out = [
        {"facet": "foundations", "question": f"definitions, taxonomy and overview of {t}",
         "papers": [(f"{t} survey", "rel"), (t, "cited")], "web": [f"{t} guide"], "survey_bonus": True},
        {"facet": "reference_works", "question": f"how the reference datasets and systems for {t} were built",
         "papers": [(f"{t} dataset", "cited"), (f"{t} benchmark dataset", "rel")]},
    ]
    for term in (terms or [])[:5]:
        term = term.strip()
        if term:
            out.append({"facet": f"concept: {term}", "question": f"what is {term} in {t}, its definition and why it "
                        "matters", "papers": [(f"{term} {t}", "rel"), (f"{term} {head}", "cited")],
                        "web": [f"{term} {head}"]})
    for c in checks:
        if not c.query:
            continue
        out.append({"facet": f"decision: {c.id}", "question": c.en,
                    "papers": [(c.query.format(t=t), "rel"), (c.query.format(t=t), "cited")]})
    out += [
        {"facet": "practice", "question": f"best practices and lessons learned for {t}",
         "papers": [(f"{t} lessons learned", "rel")], "web": [f"{t} best practices"], "videos": [f"{t} tutorial"]},
        {"facet": "pitfalls", "question": f"common mistakes, failure cases and pitfalls of {t}",
         "papers": [(f"{t} challenges limitations", "rel")], "web": [f"{t} mistakes to avoid"]},
        {"facet": "tools", "question": f"tools and recent automated methods for {t}", "repos": [t],
         "papers": [(t, "recent")]},
    ]
    return out


def _openalex(query: str, mode: str) -> list[Paper]:
    from kb.research import sources

    oa = sources.OpenAlex()
    try:
        if mode == "cited":   # the canonical works: most cited among those matching every word
            q = re.sub(r"[,:|]", " ", query)
            return oa.works_filter(f"title_and_abstract.search:{q}", 15)
        return oa.search(query, 25, date.today().year - 2 if mode == "recent" else None)
    finally:
        oa.close()


def _with_own_db(fn: Any) -> Any:
    def run(*args: Any) -> Any:
        db = Database()
        try:
            return fn(db, *args)
        finally:
            db.close()
    return run


def _search_all(fs: list[dict[str, Any]]) -> tuple[dict[tuple[int, str, str], Any], dict[str, str]]:
    from kb.research.platforms import github_search
    from kb.research.web import web_search
    from kb.research.youtube import youtube_search

    from kb.research.web import has_engine

    web = has_engine()

    jobs: dict[tuple[int, str, str], Any] = {}
    Database().close()   # create / migrate once before the threads
    pool = ThreadPoolExecutor(max_workers=8)
    for i, f in enumerate(fs):
        for q, mode in f.get("papers", []):
            jobs[(i, "papers", f"{q} [{mode}]")] = pool.submit(_openalex, q, mode)
        for q in f.get("web", []) if web else []:
            jobs[(i, "web", q)] = pool.submit(_with_own_db(lambda db, q: web_search(db, q, 6)), q)
        for q in f.get("videos", []):
            jobs[(i, "videos", q)] = pool.submit(_with_own_db(lambda db, q: youtube_search(db, q, 4)), q)
        for q in f.get("repos", []):
            jobs[(i, "repos", q)] = pool.submit(_with_own_db(lambda db, q: github_search(db, q, "repos", limit=5)), q)
    done, _ = wait(jobs.values(), timeout=SEARCH_DEADLINE)
    pool.shutdown(wait=False, cancel_futures=True)
    results, status = {}, {}
    for key, fut in jobs.items():
        name = f"{key[1]}: {key[2]}"
        if fut not in done:
            status[name] = "timeout"
            continue
        try:
            results[key] = fut.result()
            status[name] = "ok"
        except Exception as e:
            status[name] = f"failed: {type(e).__name__}"
    return results, status


def _hits(res: Any, platform: str) -> list[dict[str, Any]]:
    if not isinstance(res, dict):
        return []
    return res.get("results") or [] if isinstance(res.get("results"), list) else []


def _read(picks: list[tuple[str, str]]) -> dict[str, str]:
    """Read the chosen sources in parallel: ("paper", id) full texts, ("web", url) pages. Returns id -> status."""
    from kb.process.fulltext import fetch_papers
    from kb.research.web import fetch_url

    def one(kind: str, ref: str) -> tuple[str, str]:
        db = Database()
        try:
            if kind == "paper":
                rep = fetch_papers(db, [ref])["papers"]
                return ref, str(next(iter(rep.values()), "?"))
            r = fetch_url(db, ref)
            return r.get("id", ref), r.get("error") or f"indexed ({r.get('indexed')})"
        except Exception as e:
            return ref, f"failed: {type(e).__name__}: {str(e)[:120]}"
        finally:
            db.close()

    if not picks:
        return {}
    pool = ThreadPoolExecutor(max_workers=4)
    futs = [pool.submit(one, k, r) for k, r in picks]
    done, _ = wait(futs, timeout=READ_DEADLINE)
    pool.shutdown(wait=False, cancel_futures=False)   # unfinished reads still land in the base for later searches
    out = {}
    for (k, r), f in zip(picks, futs, strict=True):
        if f in done:
            ref, st = f.result()
            out[ref] = st
        else:
            out[r] = "still reading in the background — search_fulltext will find it in a minute"
    return out


def _read_ids(db: Database, ids: list[str]) -> set[str]:
    if not ids:
        return set()
    q = ",".join("?" * len(ids))
    return {r[0] for r in db.conn.execute(f"SELECT DISTINCT paper_id FROM chunks WHERE paper_id IN ({q})", ids)}


def readiness(db: Database, section: str, depth: str = "normal") -> dict[str, Any]:
    """Is the topic's specialist knowledge deep enough to advise? Counts items per kind and sources really read."""
    from kb import notebook

    kn = notebook.knowledge(section)
    gate = GATES.get(depth, GATES["normal"])
    counts = {k: len(v) for k, v in kn["knowledge"].items()}
    cited = {i for items in kn["knowledge"].values() for it in items for i in it["sources"]}
    ids = [((db.get_paper(i) or {"id": i})["id"]) for i in cited | set(kn["papers"])]
    read = len(_read_ids(db, ids))
    ru = _lang() == "ru"
    names = {"concept": ("понятий", "concepts"), "decision": ("решений", "decisions"),
             "practice": ("практик", "practices"), "pitfall": ("ловушек", "pitfalls")}
    missing = [f"+{gate[k] - counts.get(k, 0)} {names[k][0 if ru else 1]}" for k in names if counts.get(k, 0) < gate[k]]
    if read < gate["read"]:
        missing.append(f"+{gate['read'] - read} " + ("прочитанных источников" if ru else "sources read in full"))
    return {"section": kn["section"], "depth": depth, "knowledge_counts": counts, "sources_read": read,
            "expert_ready": not missing, "missing": missing}


def study_topic(db: Database, topic: str, task: str | None = None, terms: list[str] | None = None,
                kinds: list[str] | None = None, read: int = 10, section: str | None = None,
                depth: str = "normal") -> dict[str, Any]:
    from kb.critic.scoring import grade_and_save
    from kb.process.search import search
    from kb.research.find import _key
    from kb.research.methods import _semantic

    if not topic.strip() or not topic.isascii():
        return {"error": "pass topic as a short English phrase (e.g. 'video action annotation'); put the user's "
                         "request in `task` and the field's key terms in `terms`"}
    context = f"{topic} {task or ''} {' '.join(terms or [])}"
    profs = profiles_for(context, kinds)
    checks = checks_for(profs, context)                      # every decision the user's work has to settle
    studied = checks_for(profiles_for(topic, kinds), context)  # searched: the checklist of the field itself
    fs = facets(topic, terms, studied)
    results, status = _search_all(fs)

    # papers: scored per facet by how much of the facet's query they cover (title counts double), meaning and
    # citations; each paper lands in the one facet it fits best; papers covering under half the words are dropped
    today = date.today()
    found: dict[str, Paper] = {}
    seen: dict[str, set[int]] = {}
    for (i, platform, _), res in results.items():
        if platform != "papers":
            continue
        for p in res or []:
            k = _key(p)
            if k not in found or len(p.abstract) > len(found[k].abstract):
                found[k] = p
            seen.setdefault(k, set()).add(i)
    best: dict[str, tuple[float, int]] = {}
    twords = _words(topic)
    for i, f in enumerate(fs):
        keys = [k for k, idx in seen.items() if i in idx]
        if not keys:
            continue
        qwords = _words(" ".join(q for q, _ in f.get("papers", [])))
        sims = _semantic(f"{topic}. {f['question']}", [found[k] for k in keys])
        for k, sim in zip(keys, sims, strict=True):
            p = found[k]
            text = f"{p.title} {p.abstract}"
            cover = _cover(qwords, text)
            if cover < 0.5 or (cover < 0.6 and _cover(twords, text) < 0.34):   # off the query, or loosely on it
                continue                                                        # and off the field
            s = 0.5 * sim + 0.3 * cover + 0.2 * _cover(qwords, p.title) + 0.03 * math.log10(1 + (p.citation_count or 0))
            if f.get("survey_bonus") and SURVEY.search(p.title or ""):
                s += 0.08
            if f["facet"] == "tools" and p.published and (today - p.published).days < 400:
                s += 0.03
            if k not in best or s > best[k][0]:
                best[k] = (s, i)

    per_facet: list[list[dict[str, Any]]] = [[] for _ in fs]
    for k, (sc, i) in sorted(best.items(), key=lambda kv: kv[1][0], reverse=True):
        if sum(x["kind"] == "paper" for x in per_facet[i]) >= 3:
            continue
        pid, _ = db.upsert_paper(found[k])
        g = grade_and_save(db, pid) or {}
        p = found[k]
        per_facet[i].append({"id": pid, "kind": "paper", "title": p.title, "year": p.year, "venue": p.venue,
                             "citations": p.citation_count, "grade": g.get("grade"), "url": p.url,
                             "relevance": round(sc, 3), "_score": sc,
                             "_readable": bool(p.arxiv_id or p.pdf_url), "_arxiv": bool(p.arxiv_id)})
    db.commit()
    for (i, platform, query), res in results.items():
        if platform == "papers":
            continue
        qwords = _words(query)
        kept = [h for h in _hits(res, platform)
                if _cover(qwords, f"{h.get('title')} {h.get('snippet') or h.get('summary') or ''}") >= 0.5]
        for h in kept[:3 if platform == "web" else 2]:
            per_facet[i].append({"id": h.get("id"), "kind": h.get("kind") or "web", "title": h.get("title"),
                                 "url": h.get("url"), "grade": h.get("grade"), "_score": 0.45,
                                 "_readable": platform in ("web", "videos")})

    # read the best right away: overall ranking, foundations / reference works / concepts first, at most two per
    # facet, arXiv before publisher PDFs (often blocked), about a third web guides
    order = []
    for i, items in enumerate(per_facet):
        lead = 0.1 if fs[i]["facet"] in ("foundations", "reference_works") or fs[i]["facet"].startswith("concept") else 0
        for it in items:
            if it["_readable"]:
                order.append((it["_score"] + lead + (0.05 if it.get("_arxiv") else 0), i, it))
    order.sort(key=lambda x: x[0], reverse=True)
    picks: list[tuple[str, str]] = []
    taken: dict[int, int] = {}
    want_web = max(1, read // 3) if read else 0
    for web_pass in (True, False):   # the web share first, then papers fill the rest
        for _, i, it in order:
            is_web = it["kind"] != "paper"
            if len(picks) >= read or (web_pass and sum(k == "web" for k, _ in picks) >= want_web):
                break
            ref = it["url"] if is_web else it["id"]
            if is_web != web_pass or taken.get(i, 0) >= 2 or any(r == ref for _, r in picks):
                continue
            picks.append(("web" if is_web else "paper", ref))
            taken[i] = taken.get(i, 0) + 1
    read_status = _read(picks[:max(0, min(read, 15))])
    ok = [r for r, st in read_status.items() if st.startswith(("indexed", "already"))]
    ok_ids = {(db.get_paper(r) or {"id": r})["id"] for r in ok}

    out_facets = []
    used_chunks: set[int] = set()
    for i, f in enumerate(fs):
        passages = []
        if ok_ids:
            try:
                for h in search(db, f["question"], k=4, paper_ids=ok_ids, per_paper=1):
                    if h["chunk_id"] in used_chunks or len(passages) >= 2:
                        continue
                    used_chunks.add(h["chunk_id"])
                    passages.append({"chunk_id": h["chunk_id"], "source": h["paper_id"], "citation": h["citation"],
                                     "text": " ".join(h["text"].split())[:450]})
            except Exception as e:
                log.info("passage search failed: %s", e)
        srcs = [{k: v for k, v in it.items() if not k.startswith("_") and v not in (None, "")} for it in per_facet[i]]
        for s in srcs:
            if s.get("id") in ok_ids:
                s["read"] = True
        out_facets.append({"facet": f["facet"], "question": f["question"], "sources": srcs, "passages": passages})

    lang = _lang()
    out: dict[str, Any] = {
        "topic": topic, "task_kinds": profs, "facets": out_facets, "read": read_status,
        "decisions_to_settle": [{"id": c.id, "question": getattr(c, lang)} for c in checks],
    }
    if section:
        try:
            out["playbook"] = readiness(db, section, depth)
        except Exception as e:   # notebook errors are reported, not fatal
            out["playbook"] = {"error": str(e)}
    from kb.research.web import has_engine

    if not has_engine():
        out["web_for_your_own_search"] = [q for f in fs for q in f.get("web", [])] + [f"{topic} site:reddit.com"]
    failed = [k for k, v in status.items() if v != "ok"]
    if failed:
        out["search_problems"] = failed[:12]
    out["next"] = (
        "You are building expertise, not a reading list. 1) Read the passages; run web_for_your_own_search (if "
        "present) in your own web search for practitioner guides, tool docs and forum threads and fetch the good "
        "ones; open the strongest with "
        "search_fulltext(chunk_id=…, window=2); fetch more sources for facets with no passages (several facets "
        "per call: fetch(paper_ids=[…]) / fetch(urls=[…])), surveys and the reference-dataset papers first. "
        "2) Write what you learned with notebook_write(section, knowledge=[{kind, text, sources}]) — kinds: concept "
        "(term + definition + what it changes in the work), decision (the options, when to choose each, the "
        "trade-off), practice (what the reference works do, with numbers where there are), pitfall (failure + how "
        "to notice it), metric, tool — each in the user's language with the source ids. 3) Call study_topic again "
        "with the section until playbook.expert_ready; search_fulltext with your own English questions for gaps. "
        "4) Only then advise: explain the concepts the user needs, raise the decisions_to_settle they have not made "
        "(ask about the ones that depend on them), and propose the pipeline step by step, each step justified by a "
        "source. review_plan(plan=…) checks the user's own plan or schema.")
    return out


def review_plan(db: Database, plan: str, topic: str | None = None, section: str | None = None,
                kinds: list[str] | None = None, depth: str = "normal") -> dict[str, Any]:
    from kb import notebook

    if len(plan.strip()) < 20:
        return {"error": "pass the plan, schema, label list or pipeline as text (paste it whole)"}
    context = f"{topic or ''} {plan}"
    profs = profiles_for(context, kinds)
    checks = checks_for(profs, context)
    low = plan.lower()
    lang = _lang()
    gaps = [{"id": c.id, "question": getattr(c, lang)} for c in checks if not re.search(c.mentioned, low)]
    touched = [{"id": c.id, "question": getattr(c, lang)} for c in checks if re.search(c.mentioned, low)]
    out: dict[str, Any] = {"task_kinds": profs, "not_addressed": gaps, "addressed_check_quality": touched}
    if section:
        try:
            kn = notebook.knowledge(section)["knowledge"]
            out["learned"] = {k: kn[k] for k in ("pitfall", "practice", "decision", "metric") if kn[k]}
            tried = notebook.read_section(section)["experiments"]
            if tried:
                out["already_tried"] = tried
            out["playbook"] = readiness(db, section, depth)
        except notebook.NotebookError as e:
            out["learned"] = {"error": str(e)}
    if not out.get("learned"):
        out["warning"] = ("No studied knowledge for this topic: this is only the generic checklist. Run "
                          "study_topic(topic, section=…) first so the review rests on sources.")
    out["next"] = (
        "Review like a senior specialist: go through not_addressed and every learned pitfall / practice against the "
        "plan (and already_tried: steps that failed before are red flags); for addressed items check the plan does it well, not just mentions it. Report the 3-7 issues that "
        "most affect the result first, each with the concrete fix (rewritten labels / fields / rule / step) and the "
        "source it rests on; ask the user only what the plan cannot answer. Then give the corrected plan or schema "
        "in full.")
    return out
