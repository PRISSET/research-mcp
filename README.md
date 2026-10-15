# research-mcp

MCP-сервер для ML. Подключается к Claude Code, Codex CLI или любому другому MCP-клиенту и даёт агенту
доступ к свежим данным: статьи, GitHub, Hugging Face, форумы, веб.

Зачем: знания модели устаревают. Агент советует модели годичной давности, качает bf16 там, где хватило бы FP8, и
гадает при ошибках вместо того, чтобы найти issue с исправлением. С этим сервером он проверяет всё по живым
источникам.

Что умеет:

- показать, какие модели вышли под задачу за последние месяцы, и сравнить их (размер, контекст, лицензия,
  бенчмарки из карточек);
- подобрать веса под вашу видеокарту: bf16, FP8, INT8, GGUF;
- посчитать память и время для инференса, LoRA, QLoRA и полного fine-tune;
- найти по тексту ошибки issues и PR с исправлением и проверить, не устарели ли версии библиотек;
- найти свежие методы под конкретную задачу («сэмплер лучше res_multistep при том же числе шагов») с цифрами из
  статей и ссылкой на код;
- показать, что внутри датасета, и найти код к статье;
- перед исследованием задать уточняющие вопросы, если задача описана неполно;
- записать результаты со ссылками в один файл `RESEARCH.md`.

Ключи не нужны.

## Установка

Отправьте агенту (Claude Code или Codex):

```text
Установи MCP-сервер research: выполни
curl -fsSL https://raw.githubusercontent.com/PRISSET/research-mcp/main/scripts/install.sh | bash -s -- --agent-rules
и скажи, чем закончилось.
```

Скрипт клонирует репозиторий в `~/research-mcp`, ставит uv и зависимости и регистрирует сервер в Claude Code и
Codex. Флаг `--agent-rules` добавляет в `~/.claude/CLAUDE.md` и `~/.codex/AGENTS.md` короткое правило: проверять
модели и версии через этот сервер. После установки перезапустите агента.

Сервер работает на процессоре, видеокарта остаётся вам. При каждом запуске он сам обновляется с GitHub, если вы не
меняли файлы в папке установки.

Флаги установщика: `--notebook FILE`, `--lang ru|en`, `--no-embed` (без смыслового поиска, легче), `--dir PATH`,
`--no-register`, `--remove-agent-rules`.

## Примеры запросов

> Какие открытые модели сейчас лучше всего делают lip-sync на 8 ГБ?

> Какую версию весов Qwen3-8B качать под RTX 3060 Ti?

> Хватит ли 24 ГБ на QLoRA для Qwen3-14B с контекстом 8k и сколько займёт обучение на 200M токенов?

> vLLM падает с «Missing modality» на Qwen3-Omni, разберись.

В Claude Code есть команды `/mcp__research__research <тема>` и `/mcp__research__fix_error <ошибка>`.

## Инструменты

| Для чего | Инструменты |
|---|---|
| Уточнить задачу | `research_brief` |
| Модели | `latest_models`, `model_variants`, `compare_models` |
| Ресурсы, датасеты, код | `estimate_resources`, `dataset_info`, `paper_code` |
| Поиск | `research_sweep`, `find_methods`, `find_papers`, `github_search`, `hf_search`, `forum_search`, `web_search`, `follow_citations` |
| Чтение | `fetch_url`, `fetch_fulltext`, `get_source`, `search_fulltext`, `read_passage`, `find_paper_numbers` |
| Ошибки и версии | `diagnose_error`, `check_versions` |
| Проверка утверждений | `critique`, `verify_claim`, `what_others_say` |
| RESEARCH.md | `notebook_read`, `notebook_refresh`, `notebook_start`, `notebook_add_source`, `notebook_write` |

Источники: OpenAlex, arXiv, Semantic Scholar, GitHub, Hugging Face, Hacker News, Stack Overflow, DuckDuckGo / Bing
(или Brave / Tavily с ключом). Каждому источнику ставится оценка от A (официальный) до D (заброшенный или
спорный).

Расчёты `estimate_resources` сделаны по формулам, а не по замерам. Погрешность обычно до 15%, допущения
выводятся вместе с ответом.

## RESEARCH.md

Один файл на все исследования. Сверху оглавление, ниже по разделу на тему: вопрос, выводы и источники с
заметками. Файл можно править руками: сервер меняет только текст между метками `<!-- research:… -->`. Добавить в
него можно только источники, которые нашли инструменты, так что выдуманных ссылок не будет.

## Ручная установка

```bash
git clone https://github.com/PRISSET/research-mcp.git && cd research-mcp
uv sync --extra fulltext --extra embed
claude mcp add research -s user -e CUDA_VISIBLE_DEVICES= -- uv --directory "$(pwd)" run research-mcp
codex mcp add research --env CUDA_VISIBLE_DEVICES= -- uv --directory "$(pwd)" run research-mcp
```

Для Codex поднимите таймауты в `~/.codex/config.toml`:

```toml
[mcp_servers.research]
startup_timeout_sec = 120
tool_timeout_sec = 300
```

Другие клиенты (stdio):

```json
{ "mcpServers": { "research": { "command": "uv", "args": ["--directory", "/путь/к/research-mcp", "run", "research-mcp"],
                                "env": { "CUDA_VISIBLE_DEVICES": "" } } } }
```

## Настройки

Переменные окружения, все необязательные:

| Переменная | Что делает |
|---|---|
| `RESEARCH_NOTEBOOK` | путь к блокноту, по умолчанию `~/research/RESEARCH.md` |
| `RESEARCH_LANG` | `ru` или `en` |
| `GITHUB_TOKEN` | больше лимит запросов к GitHub |
| `BRAVE_API_KEY`, `TAVILY_API_KEY` | веб-поиск лучше, чем DuckDuckGo |
| `OPENALEX_API_KEY`, `S2_API_KEY` | больше лимит поиска статей |
| `KB_DEVICE` | `cpu` (по умолчанию) или `cuda` |
| `KB_CPU_THREADS`, `KB_CPU_THREADS_BG` | сколько ядер брать для поиска и для фоновой индексации |
| `KB_MODEL_IDLE` | через сколько секунд простоя выгружать модели из памяти (по умолчанию 900) |
| `KB_AUTO_UPDATE` | `0` отключает автообновление |
| `KB_HTTP_CACHE` | `0` отключает кеш ответов API |
| `KB_DATA_DIR` | где хранить локальную базу (по умолчанию `data/`) |

В `data/` лежит кеш: прочитанные тексты, векторы и ответы API. Его можно удалить, блокнот от него не зависит.

## Разработка

```bash
uv sync --extra fulltext --extra embed --group dev
uv run --group dev pytest -q
uv run kb update --check     # есть ли новая версия
```

Код в `src/kb/`: `research/` (поиск по платформам, модели, ML-расчёты), `process/` (тексты, векторы, гибридный
поиск), `critic/` (оценки источников), `notebook.py`, `update.py`, `mcp/server.py`.
