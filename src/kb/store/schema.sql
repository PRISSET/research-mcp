-- Sources: papers (keyed "arxiv:<id>", "doi:<doi>", "openalex:<W id>", "s2:<id>") and, with kind != 'paper',
-- every other researched source: GitHub repos/issues ("gh:…"), Hugging Face repos ("hf:…"), web pages ("web:…"),
-- forum threads ("hn:…", "so:…"). Columns `kind` and `extra` (JSON: stars, downloads, …) are added by
-- the migration in db.py so old databases upgrade in place.
CREATE TABLE IF NOT EXISTS papers (
    id                  TEXT PRIMARY KEY,
    title               TEXT NOT NULL,
    abstract            TEXT NOT NULL DEFAULT '',
    authors             TEXT NOT NULL DEFAULT '[]',   -- JSON array
    year                INTEGER,
    published           TEXT,
    updated             TEXT,
    arxiv_id            TEXT UNIQUE,
    doi                 TEXT,
    s2_id               TEXT,
    openalex_id         TEXT,
    categories          TEXT NOT NULL DEFAULT '[]',   -- JSON array
    venue               TEXT,
    url                 TEXT,
    pdf_url             TEXT,
    comment             TEXT,
    source              TEXT NOT NULL,
    is_seed             INTEGER NOT NULL DEFAULT 0,
    -- Semantic Scholar enrichment
    citation_count      INTEGER,
    influential_count   INTEGER,
    tldr                TEXT,
    publication_types   TEXT,                         -- JSON array
    s2_checked_at       TEXT,
    added_at            TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year);
CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi);

-- Full-text index over title + abstract (+ tldr), BM25 ranking.
CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
    paper_id UNINDEXED, title, abstract, tldr,
    tokenize = 'porter unicode61'
);

-- Log of collection runs, used for incremental updates.
CREATE TABLE IF NOT EXISTS collection_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    collector   TEXT NOT NULL,
    query       TEXT NOT NULL,
    topic       TEXT,                     -- unused, kept for old databases
    fetched     INTEGER NOT NULL DEFAULT 0,
    new         INTEGER NOT NULL DEFAULT 0,
    ran_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ---------------- Critic ----------------
CREATE TABLE IF NOT EXISTS paper_reviews (
    paper_id    TEXT PRIMARY KEY REFERENCES papers(id) ON DELETE CASCADE,
    or_forum    TEXT,
    or_venue    TEXT,
    status      TEXT NOT NULL,          -- accepted | rejected_or_pending | withdrawn | not_found
    checked_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS critique (
    source_id   TEXT PRIMARY KEY,       -- paper id
    kind        TEXT NOT NULL,          -- paper
    grade       TEXT NOT NULL CHECK (grade IN ('A', 'B', 'C', 'D')),
    reasons     TEXT NOT NULL,          -- JSON list
    flags       TEXT NOT NULL,          -- JSON list
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
-- ---------------- Full texts & semantic search ----------------
CREATE TABLE IF NOT EXISTS paper_fulltext (
    paper_id    TEXT PRIMARY KEY REFERENCES papers(id) ON DELETE CASCADE,
    status      TEXT NOT NULL,            -- ok | no_html | failed
    source      TEXT,                     -- arxiv-html | arxiv-pdf | oa-pdf
    n_chars     INTEGER,
    n_chunks    INTEGER,
    error       TEXT,
    fetched_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY,
    paper_id    TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    ord         INTEGER NOT NULL,
    section     TEXT NOT NULL,            -- "4 Method > 4.2 Distillation"
    kind        TEXT NOT NULL,            -- text | table | figure
    page        INTEGER,                  -- PDF source only
    text        TEXT NOT NULL,
    n_words     INTEGER NOT NULL,
    emb         BLOB,                     -- float16 vector (embedding model in meta)
    UNIQUE (paper_id, ord)
);
CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, section, content='chunks', content_rowid='id', tokenize = 'porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text, section) VALUES (new.id, new.text, new.section);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, section) VALUES ('delete', old.id, old.text, old.section);
END;

-- Quantitative claims extracted from full texts ("2.3x faster on H100 at 50 steps").
CREATE TABLE IF NOT EXISTS paper_claims (
    id          INTEGER PRIMARY KEY,
    paper_id    TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    chunk_id    INTEGER REFERENCES chunks(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL,            -- speedup | memory | steps | latency | quality_metric
    value       REAL,
    unit        TEXT,
    hardware    TEXT,                     -- JSON list of GPU names mentioned in the sentence
    models      TEXT,                     -- JSON list of model names mentioned
    sentence    TEXT NOT NULL,
    extractor   TEXT NOT NULL             -- rules-v1 | llm:<model>
);
CREATE INDEX IF NOT EXISTS idx_claims_paper ON paper_claims(paper_id);
CREATE INDEX IF NOT EXISTS idx_claims_kind ON paper_claims(kind);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Citation contexts from Semantic Scholar: what citing papers SAY about a paper (critical ones flagged).
CREATE TABLE IF NOT EXISTS citation_contexts (
    paper_id      TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    citing_s2_id  TEXT NOT NULL,
    citing_title  TEXT,
    citing_year   INTEGER,
    citing_arxiv  TEXT,
    influential   INTEGER,
    intents       TEXT,                 -- JSON list (background | methodology | result)
    context       TEXT NOT NULL,
    stance        TEXT NOT NULL,        -- critical | possible_critical | comparative | neutral
    PRIMARY KEY (paper_id, citing_s2_id, context)
);
CREATE INDEX IF NOT EXISTS idx_citctx_stance ON citation_contexts(paper_id, stance);

-- ---------------- Knowledge packs (src/kb/packs) ----------------
CREATE TABLE IF NOT EXISTS packs (
    name        TEXT PRIMARY KEY,
    version     INTEGER NOT NULL,
    items       INTEGER NOT NULL DEFAULT 0,
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
-- Which pack a document or recipe came from, with its content hash: an update rewrites only changed records.
CREATE TABLE IF NOT EXISTS pack_items (
    pack        TEXT NOT NULL,
    item_id     TEXT NOT NULL,
    type        TEXT NOT NULL,            -- doc | recipe
    hash        TEXT NOT NULL,
    PRIMARY KEY (pack, item_id)
);
CREATE INDEX IF NOT EXISTS idx_pack_items_id ON pack_items(item_id);

-- Training settings of published weights (safetensors metadata, trainer model cards).
CREATE TABLE IF NOT EXISTS recipes (
    id          TEXT PRIMARY KEY,         -- recipe:hf:<repo>/<file> | recipe:civitai:<version id>
    url         TEXT,
    title       TEXT,
    author      TEXT,
    base_model  TEXT,
    family      TEXT,                     -- flux.1-dev | sdxl | wan2.2 | qwen3 | …
    task        TEXT,                     -- character | style | concept | … (from tags and names)
    trainer     TEXT,
    method      TEXT,                     -- lora | locon | lokr | loha | full
    rank        INTEGER,
    alpha       REAL,
    lr          REAL,
    text_lr     REAL,
    optimizer   TEXT,
    scheduler   TEXT,
    steps       INTEGER,
    epochs      INTEGER,
    batch       INTEGER,                  -- per step: per device x gradient accumulation
    images      INTEGER,
    repeats     REAL,
    resolution  INTEGER,                  -- longest side
    precision   TEXT,
    extra       TEXT NOT NULL DEFAULT '{}',
    popularity  INTEGER,                  -- downloads (rounded)
    date        TEXT
);
CREATE INDEX IF NOT EXISTS idx_recipes_family ON recipes(family);
