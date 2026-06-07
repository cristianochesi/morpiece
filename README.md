MorPiece is a split-based tokenization library that incrementally chunks words into potentially meaningful morphemes. The splitting procedure consists of evaluating if the Tolerance Principle (Yang 2016) applies at every character every time an incoming word "traverses" the lexicon. 

Take the word "cats": a root "trie" (c->a->t->s) and a inflectional trie (s->t->a->c) are considered. 
In the default mode (token-based) "traversal" the lexicon means adding 1 to each node counter that is traversed both in the root trie and in the inflectional trie. If a path does not exists, it is initialized to 1. 
The type-based "traversal" modality updates the node counter only when new types are observed.
A split between "t" and "s" is pustulated if and only if both in the root trie and in the infl trie the tolerance principle is respected, that is: 
freq(t)/ln(freq(t) > freq(s) in the root trie and
freq(s)/ln(freq(s) > freq(t) in the infl trie
if this is the case, the "s" pendant (in this case just "s") is added to the root trie, under the "++" node.

At the end, all nodes that does not have a frequency above min_freq parameter are pruned.
MaxLength strategy is adopted to retrieve the tokens for each word.

Since version 1.4.* a type-based tokenization option is implemented ('type_based = true')

If the "order or acquisition" (ooa) parameter is set to True, each 100K of exposure a vocabulary is created to check splitting hypotheses postulated and the evidence needed (for research purposes)

Examples:

    import tokenizer_MorPiece as MoP

    mop = MoP.MorPiece(vocab_size=vocab_size, cutoff=cutoff, min_suffix_stems=3, min_frequency=min_frequency, ooa=ooa, use_tokenizers_lib=True)
    mop.train(text)
    mop.save('./mop_tokenizer/tokenizer.json')

    s = "test sentence"
    print("Sentence to tokenize: " + s)
    ids, tokens = mp.encode(s)
    print(ids, tokens)

Todo:

    - Multi word evaluation before splitting


RELEASE NOTES:
Changes in 1.4.4
----------------
• FIX – bogus root words beginning with an apostrophe ("'abbadessa"):
    Cause — the pre-tokeniser word pattern starts with an optional leading
    "any non-letter" character ([^\\r\\n\\p{L}\\p{N}]?).  On an Italian
    elision such as "l'abbadessa" the article's apostrophe is not consumed
    with "l"; instead it is picked up as that leading character of the NEXT
    word, yielding the token "'abbadessa" — and therefore a root-trie path
    that begins with "'".
    
•   Fix — one normaliser rule, applied before pre-tokenisation:
    an apostrophe immediately followed by a letter gets a space
    inserted after it          ( '(?=letter)  ->  "' "  )
    The apostrophe can then only stay with the word on its LEFT or stand
    alone; it can never open a word.  This is positional and language-
    agnostic — no word list, no Italian-specific logic:
      • "l'abbadessa"  -> "l'" + "abbadessa"   (clean: no leading "'")
      • "po'", "e'"    -> unchanged            (no letter follows the ')
      • English "don't"-> "don'" + "t"         (apostrophe kept on the left)
    Curly apostrophes (U+2019 / U+02BC) are first unified to ASCII "'" so
    the same rule covers them.  Word-final apostrophes are deliberately
    left intact: "po'" stays a legitimate apostrophe-final form and is
    never rewritten to an accented vowel (no "po'" -> "pò" corruption).
    Inconsistent ASCII-accent spellings ("e'" for "è") are NOT auto-
    corrected — that cannot be done simply or cross-linguistically and is
    left to optional, language-specific dataset preprocessing.

• NEW – CHILDES / CHAT speaker labels as special tokens
  (childes_speaker_tokens, default True):
    Line-initial CHAT speaker tiers ("*CHI:", "*MOT:", "*INVESTIGATOR:" …)
    are recognised and emitted as single atomic special tokens instead of
    being shredded into "*" + "chi" + ":".  Detection is pattern-based
    (line-initial "*", code adjacent to it, code adjacent to ":"), so any
    speaker code — standard three-letter or spelled-out — is handled and
    registered on the fly; a curated set of common codes is also seeded by
    default.  Markdown-style "* word:" bullets are NOT misread (the bullet
    has a space between "*" and the word).

• Registered special tokens are now skipped by the trie-building / morsplit
  passes, so a special token stays atomic rather than leaking its letters
  into the morpheme trie.

• Inherits all 1.4.3 / 1.4.2 behaviour.

Changes in 1.4.3
----------------
• NEW – save_complete_tries option (train(..., save_complete_tries=PATH)):
    Dumps the COMPLETE, un-pruned root + inflection tries to a single JSON
    file *before* __prune_weak_suffixes() and __optimize() run — i.e. before
    any min_frequency or min_suffix_stems pruning is applied.  Every node
    keeps its raw '##' frequency count and its placeholder 'IDX' marker, and
    the ++ suffix→stems map is included as well.

This snapshot is exactly what the companion trie-explorer script
    (morpiece_trie_explorer.py) consumes to visualise which branches the
    min_frequency cut will later discard, with the original frequencies
    shown on each node.
    
Pass a path string, or `True` for the default
    'tokenizer/complete_tries.json'.  The default is None (disabled): when
    the option is not used, training behaviour is byte-for-byte unchanged.

NOTE: in token-based mode (type_based=False) with ooa=True the trie is
    already thinned every 100 000 tokens by __incremental_cleaning, so
    'complete' there means 'complete as of end-of-training', not 'every node
    ever created'.

• Inherits all 1.4.2 fixes (bf removal, etc. — see below).

Changes in 1.4.2
----------------
• REMOVED – bf parameter:
    The branching-factor upper bound (`nd < bf`) has been removed from the
    Tolerance Principle check.  Empirically, it was suppressing splits at
    exactly the high-branching inflectional nodes where they should fire
    (Italian verb stems like `cant-` have 15–20+ productive continuations,
    which is the morphological *signal*, not noise).

The productivity criterion is now purely `d > m / log(m)` (with
    `m > cutoff` as a precondition).  This matches Yang's original TP
    formulation more directly: productivity is determined by whether the
    daughter count clears the tolerance threshold relative to the mother
    population, regardless of how many siblings the daughter has.

Guards against over-splitting at short prefixes (e.g. depth-1 nodes
    where every letter branches widely):
      1. The `cutoff` precondition: tiny daughter populations are rejected.
      2. The bilateral root×infl check: BOTH sides must clear TP.
      3. `min_suffix_stems`: spurious ++ suffixes deriving from too few
         distinct stems are pruned post-hoc.

Worked example — "asociale" vs "asola" (both depth-1 split candidates):
      • asociale: root TP at `a|s` usually fails because `as*` is a small
        fraction of all `a*`-initial tokens (d/m below 1/log(m)).  Even if
        the algorithm never explicitly produces `a-sociale`, at retrieval
        time the ++ fallback path produces `a` + `++sociale` via the
        independently-registered `sociale` family.
      • asola: same depth-1 failure, *plus* the infl side fails because
        words ending in `sola` are below cutoff.  Word stays unsplit (OOV
        or character-level via ++).
    These are the correct outcomes, achieved without `bf`.

• Inherits all 1.4.1 fixes:
    - OOA + type_based: read-only snapshots via __build_vocab_freq_snapshot
      (never call __incremental_cleaning in type_based mode).
    - speaker tokens + use_tokenizers_lib: replacement deferred to after
      HF preprocessing, applied at the token-stream level.
    - min_suffix_stems pruning (two-pass: IDX removal + orphan-path removal).

Unversioned maintenance changes
-------------------------------
• FIX – _preprocess_text destroyed special tokens appearing literally in the
  input.  When use_tokenizers_lib was on, a token such as "<pad>", "<mask>",
  "<s>", or "<speaker_A>" was fed through the normaliser, whose TIER 1
  punctuation rules isolate "<" and ">" ( "<pad>" -> "< pad >" ), after which
  the pre-tokeniser shredded it into "<" + "pad" + ">".  The [RSX] lookup in
  encode() could then never match it.  "*CHI:" survived only via a long,
  fragile raw-token reconstruction loop.

  Fix — _preprocess_text now splits the raw text on every registered special
  token FIRST, and sends only the non-special segments through HF
  normalisation + pre-tokenisation.  A special token therefore never reaches
  the normaliser and is preserved verbatim.  The raw-token reconstruction
  loop is removed; speaker-label detection/registration is handled up front
  by _process_speaker_labels on both code paths.

• OOA snapshot directory is now derived from a train(..., output_dir=...)
  argument and shared by both type_based and token-based modes (filename —
  vocab_type_* vs vocab_* — carries the mode distinction).
