#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_multilingual_morpiece.py
==============================

Train ONE MorPiece tokenizer over several language corpora while *reserving a
fixed token budget per language* (e.g. 16K each for eng / nld / zho), and export
it so that Chinese (whitespace-free) is tokenized properly rather than collapsing
to <unk>.

Requires the v1.4.5+ tokenizer_MorPiece.py (encode() ids/tokens alignment fix).

WHY per-language training instead of one pooled corpus
------------------------------------------------------
MorPiece enforces `vocab_size` in a single global cut in `MorPiece.__optimize`
(breadth-first, frequency-sorted siblings). Pooling all languages would (a) let
the highest-frequency language eat the budget, and (b) cross-contaminate the
Tolerance-Principle counts across languages. So we train one MorPiece per
language on its OWN statistics, cap each at its budget, then merge.

CHINESE: three things must line up for proper segmentation
----------------------------------------------------------
1. boundaries_discovery=True  (set automatically for zho): no word segmenter.
2. CJK-safe HF pre-tokenizer (`--hf_pipeline cjk_safe`, the DEFAULT): Whitespace
   (keeps Latin behaviour identical to MorPiece.save_HF) followed by a `.{1,24}`
   chunker. Without this the exported tokenizer uses a plain Whitespace
   pre-tokenizer, so a whole unspaced Han sentence becomes ONE "word"; anything
   over WordPiece's max_input_chars_per_word (100) becomes a single <unk>.
3. Character coverage (`--char_coverage_topk`, on for zho): boundaries_discovery
   never emits single-character roots (its split loop starts at length 2), so
   without coverage any sequence not matched as a multi-char morpheme is <unk>.
   We inject the top-K most frequent Han characters as BOTH a root token `X` and
   a continuation token `++X`, so WordPiece can always fall back to characters.
   Coverage is counted INSIDE the budget (coverage first, morphemes fill the
   rest), so zho still totals `budget` tokens.

Two merge modes
---------------
--mode soft  (DEFAULT, recommended for the submission tokenizer)
    Balanced allocation. Each language guaranteed up to `budget` slots; a string
    shared across languages is stored ONCE. Emits a single drop-in HF
    PreTrainedTokenizerFast directory.

--mode hard  (ablation: force eng->eng tokens, nld->nld tokens)
    Disjoint per-language ID ranges + a router (you pass the language tag at
    encode time). Native MorPiece encode already handles CJK segmentation, so
    Chinese works without the cjk_safe export; char coverage is a soft-mode
    feature and is NOT injected here. Sacrifices cross-lingual subword sharing.

Reading parquet directly from the Hub
-------------------------------------
The BabyBabelLM datasets (BabyLM-community/babylm-eng | -nld | -zho) are GATED:
log in (`huggingface-cli login`), click "Agree" on each dataset page, then a
token is required. Pass a repo id and this script downloads + materialises one
parquet via `datasets`. If you already have local parquets, pass those paths
instead (avoids gating + the offline-compute-node problem; gp02/cn02 usually
have no internet -- run any Hub download on a login node first).

Accepted --lang SOURCE per language (any of)
--------------------------------------------
  * a local directory of .txt files  (e.g. eng/ holding wikipedia.txt, gutenberg.txt)
  * a local .parquet file            (text in --text_column, default 'text')
  * an HF dataset repo id            (downloaded + materialised to one parquet)
MorPiece.train() reads a directory by concatenating every top-level file (no
recursion). The Han-character scan for zho coverage reads the same directory.

Examples
--------
# soft, directories of .txt files (your babylm_2026_clean_txt layout):
python build_multilingual_morpiece.py --mode soft --budget 16000 \
    --lang eng:../data/babylm_2026_clean_txt/eng \
    --lang nld:../data/babylm_2026_clean_txt/nld \
    --lang zho:../data/babylm_2026_clean_txt/zho \
    --zho_min_frequency 5 --char_coverage_topk 3500 \
    --output_dir ./tokenizer/mop_eng_nld_zho_16k

# soft, local parquets, default cjk_safe export + zho char coverage:
python build_multilingual_morpiece.py --mode soft --budget 16000 \
    --lang eng:../data/babylm_2026_pq/eng/all.parquet \
    --lang nld:../data/babylm_2026_pq/nld/all.parquet \
    --lang zho:../data/babylm_2026_pq/zho/all.parquet \
    --zho_min_frequency 5 --char_coverage_topk 3500 \
    --output_dir ./tokenizer/multi_eng_nld_zho_16k_soft

# soft, streaming from the Hub (needs HF login + accepted terms):
python build_multilingual_morpiece.py --mode soft --budget 16000 \
    --lang eng:BabyLM-community/babylm-eng \
    --lang nld:BabyLM-community/babylm-nld \
    --lang zho:BabyLM-community/babylm-zho \
    --output_dir ./tokenizer/multi_eng_nld_zho_16k_soft
"""

import os
import re
import json
import argparse
from collections import Counter

import tokenizer_MorPiece as MoP

# Headroom added to each language's vocab_size so we can trim to exactly `budget`
# morpheme tokens after subtracting the (variable) number of special tokens.
SPECIAL_HEADROOM = 80

# Per-language defaults. zho: boundaries_discovery (no whitespace) + char
# coverage. eng/nld: normal whitespace path. type_based=False is recommended for
# morphology (token counts drive the TP threshold; see the MorPiece docstring).
DEFAULT_LANG_CFG = {
    "zho": {"boundaries_discovery": True,  "type_based": False, "char_coverage": True},
    "_":   {"boundaries_discovery": False, "type_based": False, "char_coverage": False},
}

# Han ranges for character coverage (CJK Unified + Ext-A). CJK sentence
# punctuation we always add so '。、，！？…' etc. don't become <unk>.
_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
CJK_PUNCT = list("。、，！？；：、…—·「」『』（）《》〈〉""''")


# ---------------------------------------------------------------------------
# 1. Get a local parquet path for a language (download from the Hub if needed)
# ---------------------------------------------------------------------------
def resolve_source(source: str, lang: str, text_column: str, cache_dir: str) -> str:
    """`source` is one of: a local .parquet file, a local DIRECTORY of .txt
    files, or an HF dataset repo id. Returns something MorPiece.train() accepts
    directly (a .parquet path or a directory). HF repo ids are downloaded to one
    parquet. NOTE: MorPiece.train() reads a directory by concatenating every
    top-level file in it (no recursion, all extensions), so put wikipedia.txt /
    gutenberg.txt / ... directly under the language dir and keep non-text files out."""
    if os.path.isdir(source):
        files = [f for f in os.listdir(source)
                 if os.path.isfile(os.path.join(source, f))]
        print(f"[{lang}] using local directory of text files: {source} "
              f"({len(files)} files: {', '.join(sorted(files)[:6])}{' ...' if len(files) > 6 else ''})")
        return source
    if source.endswith(".parquet") and os.path.isfile(source):
        print(f"[{lang}] using local parquet: {source}")
        return source
    if os.path.exists(source):
        raise ValueError(f"[{lang}] '{source}' exists but is neither a .parquet "
                         f"file nor a directory.")

    print(f"[{lang}] downloading from the Hub: {source} (gated -> needs HF login + accepted terms)")
    try:
        from datasets import load_dataset, concatenate_datasets
    except ImportError:
        raise ImportError("pip install datasets  (needed to stream from the Hub)")
    token = os.environ.get("HF_TOKEN", True)   # uses cached `huggingface-cli login`
    obj = load_dataset(source, token=token)
    ds = (concatenate_datasets([obj[k] for k in obj.keys()])
          if hasattr(obj, "values") else obj)
    if text_column not in ds.column_names:
        raise ValueError(f"[{lang}] column '{text_column}' not in {ds.column_names}.")
    os.makedirs(cache_dir, exist_ok=True)
    out = os.path.join(cache_dir, f"{lang}.parquet")
    ds.select_columns([text_column]).to_parquet(out)
    print(f"[{lang}] wrote {len(ds)} rows -> {out}")
    return out


# ---------------------------------------------------------------------------
# 2. CJK character frequencies (for coverage injection)
# ---------------------------------------------------------------------------
def cjk_char_freq(source: str, text_column: str) -> Counter:
    """Count Han characters in a .parquet file OR a directory of .txt files."""
    counter = Counter()
    if os.path.isdir(source):
        for fname in sorted(os.listdir(source)):
            fpath = os.path.join(source, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        counter.update(_HAN_RE.findall(line))
            except OSError:
                continue
    else:
        import pandas as pd
        df = pd.read_parquet(source, columns=[text_column])
        for text in df[text_column].dropna().astype(str):
            counter.update(_HAN_RE.findall(text))
    return counter


# ---------------------------------------------------------------------------
# 3. Train one MorPiece for one language, capped at its budget
# ---------------------------------------------------------------------------
def train_language(source_path: str, lang: str, budget: int, text_column: str,
                   cutoff: int, min_frequency: int, min_suffix_stems: int, cfg: dict):
    print(f"\n=== training [{lang}]  budget={budget}  min_freq={min_frequency}  "
          f"bd={cfg['boundaries_discovery']}  type_based={cfg['type_based']} ===")
    mp = MoP.MorPiece(
        vocab_size=budget + SPECIAL_HEADROOM,   # trimmed to `budget` morphemes later
        cutoff=cutoff,
        min_frequency=min_frequency,
        min_suffix_stems=min_suffix_stems,
        ooa=False,
        type_based=cfg["type_based"],
        use_tokenizers_lib=True,
        boundaries_discovery=cfg["boundaries_discovery"],
    )
    mp.train(source_path, text_column=text_column)   # MorPiece.train() accepts a dir or a .parquet
    return mp


# ---------------------------------------------------------------------------
# 4. Pull specials / morphemes / coverage out of a trained MorPiece
# ---------------------------------------------------------------------------
def specials_of(mp) -> list:
    return [t for t in mp.roots["[RSX]"].keys() if t]


def union_specials(instances: list) -> list:
    ordered = ["<unk>", "<pad>", "<s>", "</s>", "<mask>"]   # pinned to IDs 0..4
    for mp in instances:
        for tok in specials_of(mp):
            if tok not in ordered:
                ordered.append(tok)
    return ordered


def morphemes_of(mp, global_specials: set, budget: int) -> list:
    """Non-special vocab, ID-ordered (MorPiece's BFS/frequency priority), <= budget."""
    return [t for t in mp.vocab_to_id.keys()
            if t and t not in global_specials][:budget]


def coverage_tokens(char_freq: Counter, topk: int) -> list:
    """Top-K Han chars as root `X` and continuation `++X`, plus CJK punctuation.
    Each char consumes two slots (first-position + continuation)."""
    toks = []
    seen = set()
    for ch in (CJK_PUNCT + [c for c, _ in char_freq.most_common(topk)]):
        for form in (ch, "++" + ch):
            if form not in seen:
                seen.add(form)
                toks.append(form)
    return toks


def lang_tokens(mp, global_specials: set, budget: int,
                char_freq: Counter = None, topk: int = 0) -> list:
    """Final per-language token list, capped at `budget`. Character coverage
    (if requested) takes priority; learned morphemes fill the remaining budget."""
    final, seen = [], set()
    if char_freq is not None and topk > 0:
        for t in coverage_tokens(char_freq, topk):
            if len(final) >= budget:
                break
            if t not in seen:
                seen.add(t); final.append(t)
    for t in morphemes_of(mp, global_specials, budget):
        if len(final) >= budget:
            break
        if t not in seen:
            seen.add(t); final.append(t)
    return final


# ---------------------------------------------------------------------------
# 5. CJK-safe HF tokenizer.json (overwrites the simple one save_HF writes)
# ---------------------------------------------------------------------------
def export_cjk_safe_tokenizer_json(vocab_to_id: dict, output_dir: str):
    from tokenizers import Tokenizer, Regex
    from tokenizers.models import WordPiece as WPModel
    from tokenizers import normalizers, pre_tokenizers, decoders, processors

    size = len(vocab_to_id)
    vocab = [""] * size
    for t, i in vocab_to_id.items():
        if 0 <= i < size:
            vocab[i] = t
    for i in range(size):
        if vocab[i] == "":
            vocab[i] = "<unk>"
    vocab_dict = {t: i for i, t in enumerate(vocab) if t.strip()}

    model = WPModel(vocab_dict, unk_token="<unk>",
                    continuing_subword_prefix="++", max_input_chars_per_word=100)
    tok = Tokenizer(model)
    tok.normalizer = normalizers.Sequence([normalizers.Lowercase(), normalizers.NFKC()])
    # Whitespace discards spaces (Latin == current save_HF behaviour); the
    # .{1,24} chunker cuts unspaced Han runs so they never blanket-<unk>.
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),
        pre_tokenizers.Split(Regex(".{1,24}"), behavior="isolated"),
    ])
    tok.decoder = decoders.WordPiece(prefix="++", cleanup=True)
    tok.post_processor = processors.TemplateProcessing(
        single="<s> $A", special_tokens=[("<s>", 2)])
    tok.add_special_tokens(["<unk>", "<pad>", "<s>", "</s>", "<mask>"])
    tok.save(os.path.join(output_dir, "tokenizer.json"))
    print(f"  CJK-safe pre-tokenizer written -> {os.path.join(output_dir, 'tokenizer.json')}")


# ---------------------------------------------------------------------------
# 6a. SOFT merge -> single drop-in HF tokenizer
# ---------------------------------------------------------------------------
def build_soft(instances, char_freqs, budget, output_dir, model_max_length,
               hf_pipeline, char_coverage_topk):
    order = list(instances.keys())
    specials_ordered = union_specials([instances[l] for l in order])
    global_specials = set(specials_ordered)

    merged = {tok: i for i, tok in enumerate(specials_ordered)}
    nid = len(specials_ordered)
    per_lang_kept, shared = {}, 0

    for lang in order:
        cf = char_freqs.get(lang)
        topk = char_coverage_topk if cf is not None else 0
        kept = 0
        for tok in lang_tokens(instances[lang], global_specials, budget, cf, topk):
            if tok in merged:
                shared += 1
            else:
                merged[tok] = nid; nid += 1
            kept += 1
        per_lang_kept[lang] = kept

    container = MoP.MorPiece(vocab_size=len(merged), ooa=False, use_tokenizers_lib=True)
    container.vocab_to_id = merged
    container.id_to_vocab = {v: k for k, v in merged.items()}
    container.idx = len(merged)
    os.makedirs(output_dir, exist_ok=True)
    container.save_HF(output_dir, model_max_length=model_max_length)   # writes all config files
    if hf_pipeline == "cjk_safe":
        export_cjk_safe_tokenizer_json(merged, output_dir)             # swap in CJK-safe tokenizer.json

    with open(os.path.join(output_dir, "multilingual_meta.json"), "w") as f:
        json.dump({
            "mode": "soft", "hf_pipeline": hf_pipeline,
            "budget_per_language": budget, "languages": order,
            "char_coverage_topk": char_coverage_topk,
            "char_coverage_languages": [l for l in order if char_freqs.get(l) is not None],
            "tokens_per_language": per_lang_kept,
            "shared_token_instances_deduped": shared,
            "n_special": len(specials_ordered), "final_vocab_size": len(merged),
        }, f, indent=2, ensure_ascii=False)

    print("\n--- SOFT merge done ---")
    print(f"  specials: {len(specials_ordered)}   pipeline: {hf_pipeline}")
    for l in order:
        cov = " (incl. char coverage)" if char_freqs.get(l) is not None else ""
        print(f"  {l}: {per_lang_kept[l]} tokens{cov}")
    print(f"  shared (deduped): {shared}")
    print(f"  FINAL vocab size: {len(merged)}  ->  {output_dir}")
    print(f"  load: PreTrainedTokenizerFast.from_pretrained('{output_dir}')")


# ---------------------------------------------------------------------------
# 6b. HARD merge -> per-language native tokenizers + router
# ---------------------------------------------------------------------------
ROUTER_SRC = r'''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""multilingual_morpiece_router.py  (auto-generated)

Forces per-language tokenization: eng text -> eng ID range, nld -> nld range,
zho -> zho range. Specials are shared at the front of the global ID space; each
language's morphemes occupy a disjoint contiguous block, so the same surface
string gets a different global ID per language. Native MorPiece encode handles
CJK segmentation directly.

    from multilingual_morpiece_router import MultilingualMorPieceRouter
    tok = MultilingualMorPieceRouter("<this_output_dir>")
    ids = tok.encode("the dogs bark", lang="eng")
    ids = tok.encode("国家学生", lang="zho")
    n   = tok.vocab_size

For HF Trainer: easiest is to pre-tokenize each document to global IDs offline
with this router (you have each doc's `language` field), then feed the ID stream.
"""
import os
import json
import tokenizer_MorPiece as MoP


class MultilingualMorPieceRouter:
    def __init__(self, directory):
        with open(os.path.join(directory, "offsets.json"), encoding="utf-8") as f:
            self.cfg = json.load(f)
        self.vocab_size = self.cfg["vocab_size"]
        self.specials_global = self.cfg["specials_global"]
        self.lang = {}
        for code, blk in self.cfg["languages"].items():
            mp = MoP.MorPiece(ooa=False, use_tokenizers_lib=True,
                              boundaries_discovery=blk["boundaries_discovery"])
            mp.from_pretrained(os.path.join(directory, code, "native"))
            self.lang[code] = {"mp": mp, "block_start": blk["block_start"],
                               "n_special": blk["n_special"]}

    def encode(self, text, lang):
        if lang not in self.lang:
            raise ValueError(f"unknown lang '{lang}'; have {list(self.lang)}")
        info = self.lang[lang]; mp = info["mp"]
        # Map from local IDs (NOT the parallel tokens list): even after the
        # v1.4.6 alignment fix, mapping by ID is the robust path.
        local_ids, _ = mp.encode(text)
        unk = self.specials_global["<unk>"]
        out = []
        for lid in local_ids:
            if lid < info["n_special"]:
                tok = mp.id_to_vocab.get(lid, "<unk>")
                out.append(self.specials_global.get(tok, unk))
            else:
                out.append(info["block_start"] + (lid - info["n_special"]))
        return out
'''


def build_hard(instances, budget, output_dir, model_max_length):
    order = list(instances.keys())
    specials_ordered = union_specials([instances[l] for l in order])
    global_specials = set(specials_ordered)
    specials_global = {tok: i for i, tok in enumerate(specials_ordered)}
    os.makedirs(output_dir, exist_ok=True)

    cursor = len(specials_ordered)
    langs_meta = {}
    for lang in order:
        mp = instances[lang]
        kept = morphemes_of(mp, global_specials, budget)   # no coverage in hard mode
        native_dir = os.path.join(output_dir, lang, "native")
        os.makedirs(native_dir, exist_ok=True)
        mp.save_pretrained(os.path.join(native_dir, "tokenizer.json"))
        mp.save_HF(os.path.join(output_dir, lang), model_max_length=model_max_length)
        langs_meta[lang] = {
            "block_start": cursor, "block_size": len(kept),
            "n_special": len(specials_of(mp)),
            "boundaries_discovery": mp.boundaries_discovery,
        }
        cursor += len(kept)

    with open(os.path.join(output_dir, "offsets.json"), "w", encoding="utf-8") as f:
        json.dump({"mode": "hard", "budget_per_language": budget,
                   "vocab_size": cursor, "specials_global": specials_global,
                   "languages": langs_meta}, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "multilingual_morpiece_router.py"), "w",
              encoding="utf-8") as f:
        f.write(ROUTER_SRC)

    print("\n--- HARD merge done ---")
    print(f"  specials (shared): {len(specials_ordered)} -> IDs 0..{len(specials_ordered)-1}")
    for l in order:
        m = langs_meta[l]
        print(f"  {l}: {m['block_size']} tokens -> IDs {m['block_start']}..{m['block_start']+m['block_size']-1}")
    print(f"  TOTAL embedding size: {cursor}  ->  {output_dir}")
    print(f"  router: {os.path.join(output_dir, 'multilingual_morpiece_router.py')}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_lang(spec: str):
    if ":" not in spec:
        raise argparse.ArgumentTypeError(f"--lang must be CODE:SOURCE, got '{spec}'")
    code, source = spec.split(":", 1)
    return code.strip(), source.strip()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lang", action="append", type=parse_lang, required=True,
                   metavar="CODE:SOURCE",
                   help="Repeatable. SOURCE = local .parquet OR HF repo id.")
    p.add_argument("--mode", choices=["soft", "hard"], default="soft")
    p.add_argument("--budget", type=int, default=16000)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--text_column", default="text")
    p.add_argument("--cutoff", type=int, default=25)
    p.add_argument("--min_frequency", type=int, default=10)
    p.add_argument("--zho_min_frequency", type=int, default=None,
                   help="Override min_frequency for zho (lower = more single-char "
                        "coverage survives). Defaults to --min_frequency.")
    p.add_argument("--min_suffix_stems", type=int, default=3)
    p.add_argument("--hf_pipeline", choices=["cjk_safe", "simple"], default="cjk_safe",
                   help="cjk_safe (default): Whitespace + .{1,24} chunker so Chinese "
                        "segments. simple: plain Whitespace (Chinese will <unk>).")
    p.add_argument("--char_coverage_topk", type=int, default=3500,
                   help="Top-K Han chars injected as root+continuation for char-"
                        "coverage languages (zho). 0 disables. Counts inside budget.")
    p.add_argument("--model_max_length", type=int, default=1024)
    p.add_argument("--cache_dir", default="./_hf_parquet_cache")
    args = p.parse_args()

    langs = dict(args.lang)
    os.makedirs(args.output_dir, exist_ok=True)

    instances, char_freqs = {}, {}
    for code, source in langs.items():
        src = resolve_source(source, code, args.text_column, args.cache_dir)
        cfg = DEFAULT_LANG_CFG.get(code, DEFAULT_LANG_CFG["_"])
        mfreq = (args.zho_min_frequency if code == "zho" and args.zho_min_frequency
                 is not None else args.min_frequency)
        instances[code] = train_language(src, code, args.budget, args.text_column,
                                          args.cutoff, mfreq, args.min_suffix_stems, cfg)
        if cfg.get("char_coverage") and args.mode == "soft" and args.char_coverage_topk > 0:
            print(f"[{code}] scanning Han character frequencies for coverage...")
            char_freqs[code] = cjk_char_freq(src, args.text_column)
            print(f"[{code}] distinct Han chars: {len(char_freqs[code])}")

    if args.mode == "soft":
        build_soft(instances, char_freqs, args.budget, args.output_dir,
                   args.model_max_length, args.hf_pipeline, args.char_coverage_topk)
    else:
        build_hard(instances, args.budget, args.output_dir, args.model_max_length)


if __name__ == "__main__":
    main()
