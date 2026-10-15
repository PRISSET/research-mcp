"""Full texts: parsing, chunking, number extraction."""

from kb.process.claims import extract
from kb.process.fulltext import MAX_WORDS, chunk_blocks
from kb.process.parse import Block, parse_arxiv_html

HTML = """<html><body><article class="ltx_document">
<h1 class="ltx_title ltx_title_document">Fast Video DiT</h1>
<div class="ltx_abstract"><h6>Abstract</h6><p>We make video diffusion transformers fast with few steps and FP8.</p></div>
<section class="ltx_section" id="S1"><h2 class="ltx_title"><span class="ltx_tag">1 </span>Introduction</h2>
<div class="ltx_para"><p>Video models are slow. We obtain a 2.4 <math alttext="\\times"></math> speedup on H100 for Wan 2.1 at 720p resolution.</p></div>
<section class="ltx_subsection"><h3 class="ltx_title"><span class="ltx_tag">1.1 </span>1.1 Setup</h3>
<div class="ltx_para"><p>All experiments run 50 sampling steps for the teacher model and 4-step generation for the student model.</p></div>
<figure class="ltx_table"><figcaption>Table 1: Results on VBench.</figcaption>
<table><tr><td>Method</td><td>VBench</td></tr><tr><td>Ours</td><td>84.1</td></tr></table></figure>
</section></section>
<section class="ltx_bibliography"><h2>References</h2><div class="ltx_para"><p>[1] Some paper that must not be indexed at all.</p></div></section>
<section class="ltx_section"><h2 class="ltx_title"><span class="ltx_tag">5 </span>Acknowledgements</h2>
<div class="ltx_para"><p>We thank everyone who helped with this project a lot.</p></div></section>
</article></body></html>"""


def test_parse_html_sections_math_tables():
    title, blocks = parse_arxiv_html(HTML)
    assert title == "Fast Video DiT"
    sections = [b.section for b in blocks]
    assert sections[0] == "Abstract"
    assert "1 Introduction > 1.1 Setup" in sections          # duplicated "1.1" removed, path kept
    intro = next(b for b in blocks if b.section == "1 Introduction")
    assert "$\\times$" in intro.text                          # math kept as LaTeX
    table = next(b for b in blocks if b.kind == "table")
    assert "Ours | 84.1" in table.text and table.text.startswith("Table 1")
    assert not any("must not be indexed" in b.text or "thank everyone" in b.text for b in blocks)


def test_chunking_respects_sections_and_size():
    blocks = [Block("1 Intro", "text", "word " * 200), Block("1 Intro", "text", "word " * 200),
              Block("2 Method", "text", "word " * 50), Block("2 Method", "table", "Table 1: x | y"),
              Block("3 Long", "text", ("This is a sentence of eight words here. " * 150))]
    chunks = list(chunk_blocks(blocks))
    assert all(len(t.split()) <= MAX_WORDS for _, kind, t, _ in chunks if kind == "text")
    assert [c[0] for c in chunks][:2] == ["1 Intro", "2 Method"]      # 400 words fit one chunk; sections never merge
    assert ("2 Method", "table", "Table 1: x | y", None) in chunks
    assert sum(1 for c in chunks if c[0] == "3 Long") >= 3


def test_claim_extraction_keeps_conditions():
    claims = extract("We obtain a 2.4 $\\times$ speedup on H100 for Wan 2.1 at 720p resolution. "
                     "The student uses 4-step generation. As shown in [19], the FID of 12.5 is reported.")
    speed = next(c for c in claims if c["kind"] == "speedup")
    assert speed["value"] == 2.4 and "H100" in speed["hardware"] and speed["models"]
    assert any(c["kind"] == "steps" and c["value"] == 4 for c in claims)
    assert not any(c["kind"] == "speedup" and c["value"] < 1.05 for c in claims)


def test_claims_cover_other_fields_and_skip_parameters():
    found = extract("We achieve a 2-2.5x decoding speedup in a distributed setup. "
                    "Our model reaches an accuracy of 87.3 on MMLU with Llama-2. The WER of 4.2 is achieved with Whisper.")
    kinds = {(c["kind"], c["value"], c["unit"]) for c in found}
    assert ("speedup", 2.5, "x") in kinds
    assert ("quality_metric", 87.3, "ACCURACY") in kinds and ("quality_metric", 4.2, "WER") in kinds
    assert extract("HumanEval was sampled with p=0.95 and temperature 0.8 for all runs.") == []
