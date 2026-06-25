__version__ = "1.4.5"
__author__ = "Cristiano Chesi & NeTS Lab @ IUSS (with Claude Sonnet 4.6 / Opus 4.8 fixes)"
__email__ = "cristiano.chesi@iusspavia.it"
__status__ = "Research"
__date__ = "2026-06-20"
__license__ = "MIT"

"""
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

NOTE: by default (char_coverage=0.99) the most frequent characters are added to the final vocabulary at the end of training, each in both its root ("c") and inflectional ("++c") form, up to 99% of character occurrences. This gives one token per glyph (ideal for CJK — "中文" -> "中" "++文"), works identically in the exported HuggingFace WordPiece tokenizer, and leaves the WordPiece segmentation of known words unchanged. The added tokens are baked into the saved tokenizer, so char_coverage need not be set when loading a trained tokenizer.

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

Reference:

    https://github.com/cristianochesi/morpiece
"""

import os
import re
import json
from collections import deque
from math import log
from tokenizers import pre_tokenizers, decoders, normalizers, Regex, processors


class MorPiece:
    """MorPiece incrementally chunks words into potentially meaningful morphemes."""

    ids: list[int]
    tokens: list[str]
    vocab_size: int

    # -------------------------------------------------------------------------
    # CHILDES / CHAT speaker codes (v1.4.4)
    # -------------------------------------------------------------------------
    # Common line-initial CHAT speaker tiers.  Each entry C becomes the atomic
    # special token "*C:".  This is only a SEED set for cross-run consistency:
    # any speaker code actually found in the corpus — standard three-letter or
    # spelled-out, e.g. "*INVESTIGATOR:" — is detected and registered on the
    # fly during preprocessing, so the list need not be exhaustive.
    CHILDES_SPEAKER_CODES = (
        "CHI", "MOT", "FAT", "INV", "EXP", "GRA", "GRM", "GRF",
        "SIS", "BRO", "SIB", "ADU", "TEA", "CAR", "BAB", "AUN",
        "UNC", "COU", "FRI", "NEI", "OPE", "PAR", "UNK", "NON", "TOY",
        "CHILD", "MOTHER", "FATHER", "INVESTIGATOR", "EXPERIMENTER",
        "TEACHER", "ADULT", "GRANDMOTHER", "GRANDFATHER",
    )

    def __init__(
        self,
        vocab_size=30000,
        min_frequency=10,
        cutoff=100,
        special_tokens=None,
        ooa=True,
        use_tokenizers_lib=True,
        type_based=True,
        use_speaker_tokens=True,
        childes_speaker_tokens=True,
        min_suffix_stems=3,
        ooa_type_interval=1000,
        ooa_token_interval=100000,
        counter_unit='tokens',
        boundaries_discovery=False,
        char_coverage=0.99,
    ):
        """
        Parameters
        ----------
        vocab_size : int
            Maximum vocabulary size (default 30000).
        min_frequency : int
            Minimum token frequency to survive final pruning (default 10).
        cutoff : int
            Minimum mother-node frequency before TP is evaluated.  Nodes below
            this threshold are ignored (default 100).  Acts as the principal
            guard against splits at low-evidence positions.
        ooa : bool
            Save Order-of-Acquisition vocabulary snapshots (default True).
        use_tokenizers_lib : bool
            Use HuggingFace tokenizers for normalisation and pre-tokenisation.
        type_based : bool
            Update trie counts once per type instead of once per token.

            ⚠ STRUCTURAL LIMITATION for morphologically rich languages:
            In type_based mode, trie counts equal the number of distinct types
            sharing a path prefix — NOT token frequencies.  For Italian with
            ~200 K types, paradigm-segment counts (e.g. "cantav-") are only
            5–15 types.  The TP threshold tp = m / log(m) sits at ~3–7, and
            the daughter count d must exceed tp.  This barely succeeds at best.

            In TOKEN-BASED mode (type_based=False) the same segment sees
            5 000–20 000 tokens across all its forms; the threshold sits at
            ~1 000–1 500 and is easily cleared.  For Italian morphological
            tokenization, type_based=False is strongly recommended.

        use_speaker_tokens : bool
            Detect and replace line-initial "A:"–"E:" labels with <speaker_X>
            special tokens.  Mid-word labels ("INCIDAZIONE:") are left untouched.
        childes_speaker_tokens : bool
            Detect line-initial CHILDES / CHAT speaker tiers ("*CHI:", "*MOT:",
            "*INVESTIGATOR:" …) and keep each one as a single atomic special
            token instead of letting the pre-tokeniser shred it into
            "*" + "chi" + ":".  Detection is pattern-based and registers any
            speaker code it meets; CHILDES_SPEAKER_CODES seeds the common ones.
            Default True — harmless on non-CHAT corpora, where the line-initial
            "*code:" pattern simply does not occur.
        min_suffix_stems : int
            Minimum distinct root stems a ++ suffix must derive from (default 3).
            Prunes coincidental character overlaps (e.g. "nto" from only "lento").
        ooa_type_interval : int
            Types between OOA snapshots in type_based mode (default 1000).
        ooa_token_interval : int
            Amount of `counter_unit` between OOA snapshots / incremental-cleaning
            passes in TOKEN-based mode (and in --boundaries_discovery).  Default
            100000.  Previously hard-coded to 100000 tokens.
        counter_unit : str
            Unit in which token-based OOA exposure is accumulated:
            'tokens' (default; one increment per processed token — exactly the
            previous behaviour), 'chars' (UTF-8 characters), or 'syllables'
            (approximate vowel-group / per-CJK-glyph count).  The syllable is the
            most cross-linguistically comparable measure of input quantity
            (Räsänen et al. 2019, 2021), so for EN-vs-IT parity prefer
            counter_unit='syllables' over a fixed word-token interval.  Applies
            to the token-based and boundary-discovery snapshot schedules only;
            type_based snapshots remain measured in types via ooa_type_interval.
        special_tokens : list or None
            Reserved tokens; defaults to the standard set.
        """
        if special_tokens is None:
            special_tokens = ['<unk>', '<pad>', '<s>', '</s>', '<mask>', '<sep>', '<cls>']

        self.use_speaker_tokens = use_speaker_tokens
        self.speaker_token_map = (
            {L: f'<speaker_{L}>' for L in 'ABCDE'}
            if use_speaker_tokens else {}
        )
        if use_speaker_tokens:
            for st in self.speaker_token_map.values():
                if st not in special_tokens:
                    special_tokens = list(special_tokens) + [st]

        # --- v1.4.4: CHILDES / CHAT speaker tiers as atomic special tokens ---
        self.childes_speaker_tokens = childes_speaker_tokens
        if childes_speaker_tokens:
            special_tokens = list(special_tokens)
            for code in self.CHILDES_SPEAKER_CODES:
                tok = f"*{code}:"
                if tok not in special_tokens:
                    special_tokens.append(tok)

        self.special_tokens   = special_tokens
        self.unk_token_id     = 0
        self.pad_token_id     = 1
        self.bos_token_id     = 2
        self.eos_token_id     = 3
        self.mask_token_id    = 4
        self.sep_token_id     = 5
        self.cls_token_id     = 6
        self.start_of_text_symbol = '<s>'
        self.reserved_keys    = {'[RSX]', '##', 'IDX', '++'}

        self.vocab_size          = vocab_size
        self.min_frequency       = min_frequency
        self.ooa                 = ooa
        self.ooa_split           = {}
        self.ooa_type_interval   = ooa_type_interval
        if counter_unit not in ('tokens', 'chars', 'syllables'):
            raise ValueError(
                f"counter_unit must be 'tokens', 'chars' or 'syllables', "
                f"got {counter_unit!r}"
            )
        self.ooa_token_interval  = ooa_token_interval
        self.counter_unit        = counter_unit
        self.use_tokenizers_lib  = use_tokenizers_lib
        self.type_based          = type_based
        self.min_suffix_stems    = min_suffix_stems
        self.suffix_stems: dict[str, set] = {}

        self.roots  = {'[RSX]': {}, '++': {}}
        self.infls  = {}
        # --- v1.4.5 (--boundaries_discovery) ---------------------------------
        # When True, MoP does NOT pre-tokenise on whitespace.  The corpus is cut
        # only at *unambiguous* edges (punctuation + line breaks = true starts
        # "[[" / true ends "]]"); each anchored sequence traverses the root trie
        # from its true start and the infl trie from its true end, and boundaries
        # "||" are placed incrementally as soon as the bilateral TP licenses them
        # (sufficiency).  Spaces are demoted to an ordinary soft-cue symbol so
        # word boundaries re-emerge from statistics rather than being given.
        self.boundaries_discovery = boundaries_discovery
        self.SPACE_MARK = '\u2581'                 # soft space cue (HF-friendly)
        # v1.4.5: at the end of training, add the most frequent characters as
        # single-character vocab tokens (root "c" + infl "++c") up to this
        # fraction of character occurrences.  One token per glyph — far better
        # than a byte fallback for CJK.  Default 0.99; set 0/None to disable.
        # See _add_frequent_chars.  Irrelevant at load (the tokens are saved).
        self.char_coverage = char_coverage
        self._anchor_re = re.compile(r"[.!?;:,…·。！？；：，、（）()\[\]{}\"'«»\n\r\t]+")
        self.types  = {}
        self.idx    = 0
        self.ids    = []
        self.tokens = []
        self.cutoff = cutoff

        self.num_tokens_in_corpus        = 0
        self._ooa_exposure               = 0   # running token position for the OOA event log
        self.num_chars_in_corpus         = 0
        self.num_chars_in_trie           = 0
        self.num_chars_in_optimized_trie = 0

        # --- special-token splitter cache (see _special_token_splitter) ------
        self._split_re      = None
        self._split_re_n    = -1
        self._special_set   = set()

        self.set_special_tokens(self.special_tokens)

        # m_daughters is still tracked for OOA diagnostic output (it tells us
        # how branchy each node is, which is useful for parameter tuning) —
        # it just no longer participates in the TP decision as of 1.4.2.
        self.ooa_data = [['Word', 'Piece',
                          'Root Trie - Mother freq', 'Root Trie - Daughter freq',
                          'Root Trie - Daughter # of children',
                          'Infl Trie - Mother freq', 'Infl Trie - Daughter freq',
                          'Infl Trie - Daughter # of children',
                          'Vocabulary size',
                          'Token exposure']]
        self.vocab_to_id   = None
        self.vocab_to_freq = None
        self.id_to_vocab   = None
        self.current_world = ""
        self.node          = {}
        self.m_freq        = 1
        self.d_freq        = 1
        self.m_daughters   = 1

        if self.use_tokenizers_lib:
            self._init_tokenizers_components()

    # =========================================================================
    # HF tokenizers initialisation
    # =========================================================================

    def _init_tokenizers_components(self):
        self.normalizer = normalizers.Sequence([
            normalizers.Lowercase(),
            normalizers.Prepend(" "),
            normalizers.NFKC(),
            # --- v1.4.4: two-tier punctuation strategy -----------------------
            # (1) Unify typographic apostrophes (U+2019 ' , U+02BC ʼ) to ASCII
            #     U+0027 so Tier 2 sees a single codepoint.
            normalizers.Replace(Regex("[\u2019\u02bc]"), "'"),
            # (2) TIER 1 — hard-isolate every special character EXCEPT the
            #     apostrophe (a "special char" = not whitespace, letter, digit,
            #     or apostrophe).  Two zero-width insertions, because
            #     normalizers.Replace does literal replacement (no $1 backrefs):
            #       2a — space BEFORE such a char (if not already space-led)
            #       2b — space AFTER  such a char (if not already space-followed)
            #     Result: each is its own token, separated from BOTH neighbours.
            #       "house!!!"   -> "house ! ! !"
            #       "..@%$£-!!!" -> ". . @ % $ £ - ! ! !"
            #       "(parola)"   -> "( parola )"
            normalizers.Replace(Regex("(?<=\\S)(?=[^\\s\\p{L}\\p{N}'])"), " "),
            normalizers.Replace(Regex("(?<=[^\\s\\p{L}\\p{N}'])(?=\\S)"), " "),
            #     TIER 2 — the apostrophe is deliberately NOT touched here.
            #     It is kept word-internal (see pre-tokenizer) so the morpheme
            #     boundary is set by BPE statistics, not by a positional rule
            #     that would be linguistically wrong in some language (Italian
            #     elides left: l', d'; English contracts right & irregularly:
            #     do|n't, I|'m).  Word-final apostrophes ("po'", "dogs'") are
            #     covered too — no letter follows, nothing to split.
            normalizers.Replace(Regex("\n"), '\n '),
            normalizers.Replace(Regex(" *\n"), '\n'),
        ])
        self.pre_tokenizer = pre_tokenizers.Sequence([
            # TIER 2 cont. — U+0027 is added to the letter classes (the trailing
            # ' inside each [...] below) so an apostrophe flanked by letters does
            # NOT break a word: "l'acqua", "don't", "qu'il" stay ONE pre-token,
            # letting MoP learn the language-specific clitic/elision split.
            pre_tokenizers.Split(
                Regex(
                    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}']*"
                    "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}']+|[^\\r\\n\\p{L}\\p{N}]?"
                    "[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}']+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}']*"
                    "| ?\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                ),
                behavior="isolated",
                invert=False,
            ),
            pre_tokenizers.Split(
                Regex(".{1,24}"),
                behavior="isolated",
                invert=False,
            ),
        ])
        self.decoder = decoders.Sequence([
            decoders.Strip(' ', 1, 0),
            decoders.Replace("\n ", "\n"),
        ])
        self.post_processor = processors.TemplateProcessing(
            single=f"{self.start_of_text_symbol} $A",
            pair=f"{self.start_of_text_symbol} $A {self.start_of_text_symbol} $B",
            special_tokens=[(self.start_of_text_symbol, self.bos_token_id)],
        )

    # =========================================================================
    # Speaker-label preprocessing
    # =========================================================================

    _SPEAKER_RE = re.compile(r'(?m)^[ \t]*([A-Ea-e]):[ \t]*')

    # v1.4.4: line-initial CHILDES / CHAT speaker tier, e.g. "*CHI:" / "*MOT:"
    # / "*INVESTIGATOR:".  The "*" and the code are adjacent, and the code and
    # the ":" are adjacent — this is what distinguishes a real speaker tier
    # from a markdown bullet such as "* word:".
    _CHILDES_SPEAKER_RE = re.compile(r'(?m)^[ \t]*\*([A-Za-z]{2,20}):')

    def _register_special_token(self, tok: str) -> None:
        """Register a token as an atomic special token (RSX + special list)."""
        if tok not in self.roots['[RSX]']:
            self.roots['[RSX]'][tok] = {'IDX': self.idx}
            self.idx += 1
        if tok not in self.special_tokens:
            self.special_tokens.append(tok)

    def _process_speaker_labels(self, text: str) -> str:
        """Replace line-initial speaker labels with atomic special tokens.

        Handles the generic "A:"–"E:" dialogue labels (-> <speaker_X>) and,
        when childes_speaker_tokens is on, CHILDES / CHAT tiers "*CODE:"
        (kept verbatim, uppercased).  Used on BOTH the tokenizers and the
        non-tokenizers code path: it runs on the raw text, before any HF
        normalisation, and registers any speaker code it meets on the fly.
        """
        if self.childes_speaker_tokens:
            def _childes(m: re.Match) -> str:
                tok = '*' + m.group(1).upper() + ':'
                self._register_special_token(tok)
                return ' ' + tok + ' '
            text = self._CHILDES_SPEAKER_RE.sub(_childes, text)

        if not self.use_speaker_tokens:
            return text

        def _replace(m: re.Match) -> str:
            letter = m.group(1).upper()
            token  = self.speaker_token_map.get(letter)
            return (token + ' ') if token else m.group(0)

        return self._SPEAKER_RE.sub(_replace, text)

    # =========================================================================
    # Text preprocessing
    # =========================================================================

    def _special_token_splitter(self):
        """Return a compiled regex that splits text on any registered special token.

        Longest-token-first alternation, so a longer token wins over a shorter
        prefix of it (e.g. "*INVESTIGATOR:" is preferred over a hypothetical
        shorter "*INV:").  The capturing group means re.split() keeps the
        special tokens as separate elements in its output.

        The regex is cached and rebuilt only when the special-token set
        changes — which can happen during early training, as
        _process_speaker_labels registers new speaker codes on the fly.
        Returns None if no special tokens are registered.
        """
        specials = [t for t in self.roots['[RSX]'] if t]
        if self._split_re_n != len(specials):
            specials.sort(key=len, reverse=True)
            self._special_set = set(specials)
            self._split_re = (
                re.compile('(' + '|'.join(re.escape(s) for s in specials) + ')')
                if specials else None
            )
            self._split_re_n = len(specials)
        return self._split_re

    def _preprocess_text(self, text: str) -> str:
        """Full preprocessing pipeline.

        Special tokens are preserved verbatim.  Speaker labels are detected
        and registered on the RAW text by _process_speaker_labels; the text is
        then split on every registered special token, and ONLY the non-special
        segments are sent through HF normalisation + pre-tokenisation.  A
        special token can therefore never be lowercased, NFKC-folded, or
        shredded by the TIER 1 punctuation rules ("<pad>" -> "< pad >",
        "*CHI:" -> "* chi :").
        """
        # Speaker-label detection + on-the-fly registration runs on raw text,
        # for both code paths.
        text = self._process_speaker_labels(text)

        if not self.use_tokenizers_lib:
            return text

        split_re = self._special_token_splitter()
        segments = split_re.split(text) if split_re else [text]

        out = []
        for seg in segments:
            if not seg:
                continue
            if seg in self._special_set:
                # Verbatim — a special token bypasses normalisation entirely;
                # a special token is also a hard word boundary.
                out.append(seg)
            else:
                normalized = self.normalizer.normalize_str(seg)
                out.extend(tok for tok, _ in
                           self.pre_tokenizer.pre_tokenize_str(normalized))
        return " ".join(out)

    def _postprocess_tokens(self, token_ids: list) -> list:
        if not self.use_tokenizers_lib:
            return token_ids
        if token_ids and token_ids[0] != self.bos_token_id:
            return [self.bos_token_id] + token_ids
        return token_ids

    # =========================================================================
    # Core trie operations
    # =========================================================================

    def __morsplit(self, w: str) -> None:
        reversed_w = w[::-1]
        for i in range(2, len(w) + 1):
            self.__find_path(reversed_w[:len(w) + 2 - i], self.infls)
            i_m, i_d, i_nd = self.m_freq, self.d_freq, self.m_daughters
            infls_tp = self.__check_tp(self.m_freq, self.d_freq)

            self.__find_path(w[:i], self.roots)
            r_m, r_d, r_nd = self.m_freq, self.d_freq, self.m_daughters
            roots_tp = self.__check_tp(self.m_freq, self.d_freq)

            if roots_tp and infls_tp:
                stem   = w[:i - 1]
                suffix = w[i - 1:]

                if suffix not in self.suffix_stems:
                    self.suffix_stems[suffix] = set()
                self.suffix_stems[suffix].add(stem)

                if self.ooa:
                    key = self.current_world + " " + stem + "-" + suffix
                    if key in self.ooa_split:
                        self.ooa_split[key] += 1
                    else:
                        self.ooa_data.append([
                            self.current_world, stem + "-" + suffix,
                            r_m, r_d, r_nd, i_m, i_d, i_nd,
                            len(self.types), self._ooa_exposure,
                        ])
                        self.ooa_split[key] = 1

                if 'IDX' not in self.node:
                    self.node['IDX'] = 1
                self.__build_trie(suffix, self.roots['++'])

    def __build_trie(self, wordpiece: str, root: dict) -> None:
        node = root
        for ch in wordpiece:
            if ch in node:
                node[ch]['##'] += 1
            else:
                node[ch] = {'##': 1}
            node = node[ch]
        if 'IDX' not in node:
            node['IDX'] = 1

    @staticmethod
    def __incremental_cleaning(trie, freq):
        """Remove low-frequency pendant nodes (token-based mode only)."""
        queue = deque([(None, None, trie)])
        while queue:
            parent, key, current = queue.popleft()
            if isinstance(current, dict):
                if key == "[RSX]":
                    continue
                if (key != "++") and (current.get("##", 0) <= freq):
                    if parent is not None:
                        del parent[key]
                        continue
                for k, v in list(current.items()):
                    if isinstance(v, dict):
                        queue.append((current, k, v))

    def __optimize(self, trie) -> None:
        queue = deque([(None, None, trie)])
        while queue:
            parent, key, current = queue.popleft()
            if isinstance(current, dict):
                if key == "[RSX]":
                    continue
                if (key != "++") and (current.get("##", 0) <= self.min_frequency):
                    if parent is not None:
                        del parent[key]
                        continue
                if current.get("##", None) is not None:
                    del current["##"]
                if "IDX" in current:
                    if self.idx < self.vocab_size:
                        current["IDX"] = self.idx
                        self.idx += 1
                    else:
                        del parent[key]
                for k, v in list(current.items()):
                    if isinstance(v, dict):
                        queue.append((current, k, v))
        self.__build_vocab_lookup()

    def __find_path(self, word: str, trie: dict) -> None:
        node = trie
        for ch in word:
            self.node        = node[ch]
            self.m_freq      = self.d_freq
            self.d_freq      = node[ch]['##']
            self.m_daughters = len(node[ch])
            node             = node[ch]

    def __retrieve(self, string: str, trie: dict) -> None:
        node = trie
        for ch in string:
            if ch in node:
                self.tokens[-1] += ch
                if 'IDX' in node[ch]:
                    self.ids[-1] = node[ch]['IDX']
                node = node[ch]
            elif ch in self.roots['++']:
                suffix_node = self.roots['++'][ch]
                self.tokens.append('++' + ch)
                self.ids.append(suffix_node.get('IDX', 0))
                node = suffix_node
            else:
                self.ids.append(self.roots['[RSX]']['<unk>']['IDX'])
                self.tokens.append('<unk>')
                break

    def __check_tp(self, m, d) -> bool:
        """
        Tolerance Principle check (Yang 2016) — v1.4.2 form.

        Returns True iff:
          (a) mother frequency m exceeds the cutoff, AND
          (b) daughter count d strictly exceeds the tolerance threshold
              tp = m / log(m), AND
          (c) d != m  (the daughter does not exhaust the mother — i.e.,
              there is at least some alternative continuation, otherwise
              there is nothing to be productive *over*).

        The branching-factor filter (`nd < bf`) of versions ≤1.4.1 has
        been removed: high-branching nodes are exactly the inflectional
        decision points where productivity should be evaluated, not
        suppressed.  The cutoff plus the bilateral root×infl requirement
        plus min_suffix_stems already guards against spurious splits at
        low-evidence prefixes.
        """
        if m <= self.cutoff:
            return False
        tp = m / log(m)
        return (m != d) and (d > tp)

    def _add_frequent_chars(self) -> None:
        """Ensure the most frequent characters are in the vocabulary as single
        tokens, up to `char_coverage` of character occurrences (default 0.99).

        Rationale (v1.4.5): rather than a byte fallback (which shreds each CJK
        glyph into 3 UTF-8 byte tokens), we add the actual *characters* — one
        token per glyph.  For every frequent character we make sure BOTH forms
        exist:
          • root form          "c"   (word-initial)
          • inflectional form  "++c" (word-internal / continuation)
        so e.g. "中文你好" tokenises as ["中", "++文", "++你", "++好"] — and,
        because these are ordinary single-character vocab entries, the SAME
        thing happens in the exported HuggingFace WordPiece tokenizer (no model
        change, no fragmentation).

        Character frequencies are token-weighted over the (alphabetic) word
        types, so coverage is measured over real character occurrences.  This
        runs AFTER __optimize, so it only ADDS vocabulary entries and never
        changes how already-known words are segmented (a word that contains
        none of the newly-added characters is untouched; for a word that does,
        the deeper greedy match still wins, so Latin segmentation — and the
        SIGMORPHON behaviour — is unchanged).  The added tokens are written into
        self.roots, so they are saved with the tokenizer and present on reload.

        Controlled by `char_coverage` (default 0.99): set to 0/None to disable.
        Characters beyond the coverage threshold (the rare <1% tail) are left
        out and remain "<unk>".
        """
        cov = self.char_coverage
        if not cov or cov <= 0:
            return
        if not self.types:
            # no per-type counts available (e.g. boundaries_discovery) -> nothing to do
            return

        # token-weighted character frequencies over the alphabetic word types
        char_freq: dict[str, int] = {}
        for word, count in self.types.items():
            for ch in word:
                char_freq[ch] = char_freq.get(ch, 0) + count
        total = sum(char_freq.values())
        if total == 0:
            return

        target = cov * total
        ranked = sorted(char_freq.items(), key=lambda kv: kv[1], reverse=True)

        def _ensure_idx(container: dict, key: str) -> bool:
            """Give `container[key]` an IDX if it lacks one; return True if added."""
            node = container.get(key)
            if isinstance(node, dict):
                if 'IDX' in node:
                    return False
                node['IDX'] = self.idx
            else:
                container[key] = {'IDX': self.idx}
            self.idx += 1
            return True

        added_root = added_infl = n_chars = 0
        cum = 0
        for ch, f in ranked:
            if cum >= target:
                break
            cum += f
            n_chars += 1
            if ch not in self.vocab_to_id:                 # root form  "c"
                added_root += _ensure_idx(self.roots, ch)
            if ('++' + ch) not in self.vocab_to_id:        # infl form  "++c"
                added_infl += _ensure_idx(self.roots['++'], ch)

        self.__build_vocab_lookup()
        if added_root or added_infl:
            print(f"  Added frequent-char coverage to {cov:.0%}: "
                  f"+{added_root} root, +{added_infl} ++ tokens "
                  f"(covering {n_chars} distinct chars).")

    def __build_vocab_lookup(self) -> None:
        self.vocab_to_id = {}

        def traverse(trie, path):
            for k, v in trie.items():
                if k == 'IDX':
                    self.vocab_to_id[''.join(path)] = v
                elif isinstance(v, dict):
                    traverse(v, path + [k])

        traverse(self.roots, [])
        self.vocab_to_id = dict(sorted(self.vocab_to_id.items(), key=lambda x: x[1]))
        self.vocab_to_id = {k.replace("[RSX]", ""): v for k, v in self.vocab_to_id.items()}
        self.id_to_vocab = {v: k for k, v in self.vocab_to_id.items()}

    def __build_vocab_freq(self) -> None:
        """Build vocab_to_freq from trie nodes that have BOTH IDX and ## (in-training state)."""
        vocab_to_freq = {}

        def traverse(trie, path):
            if isinstance(trie, dict) and 'IDX' in trie and '##' in trie:
                vocab_to_freq[''.join(path)] = trie['##']
            for k, v in trie.items():
                if k not in ('IDX', '##') and isinstance(v, dict):
                    traverse(v, path + [k])

        traverse(self.roots, [])
        self.vocab_to_freq = dict(sorted(vocab_to_freq.items(), key=lambda x: x[1], reverse=True))

    def __build_vocab_freq_snapshot(self) -> dict:
        """
        Return a frequency-filtered vocabulary snapshot WITHOUT modifying self.roots.

        Used exclusively by the type_based OOA loop.  Unlike __build_vocab_freq
        (which sets self.vocab_to_freq as a side-effect) this method returns a
        new dict and leaves the trie completely unchanged — essential in
        type_based mode where each word form is traversed only once and any
        destructive cleaning would permanently corrupt the trie.
        """
        snapshot = {}

        def traverse(trie, path):
            if isinstance(trie, dict) and 'IDX' in trie and '##' in trie:
                freq = trie['##']
                if freq > self.min_frequency:
                    snapshot[''.join(path)] = freq
            for k, v in trie.items():
                if k not in ('IDX', '##') and isinstance(v, dict):
                    traverse(v, path + [k])

        traverse(self.roots, [])
        return dict(sorted(snapshot.items(), key=lambda x: x[1], reverse=True))

    # def __sort_trie_by_freq(self, d):
    #    if not isinstance(d, dict):
    #        return d
    #    sorted_items = sorted(
    #        d.items(),
    #        key=lambda item: item[1].get('##', float('-inf')) if isinstance(item[1], dict) else float('-inf'),
    #        reverse=True,
    #    )
    #    d.clear()
    #    for k, v in sorted_items:
    #        d[k] = self.__sort_trie_by_freq(v)
    #    return d
        
    def __sort_trie_by_freq(self, d):
        if not isinstance(d, dict):
            return d
        stack = [d]
        while stack:
            node = stack.pop()
            items = sorted(
                node.items(),
                key=lambda item: item[1].get('##', float('-inf')) if isinstance(item[1], dict) else float('-inf'),
                reverse=True,
            )
            keys = list(node.keys())
            for k in keys:
                del node[k]
            for k, v in items:
                node[k] = v
                if isinstance(v, dict):
                    stack.append(v)
        return d

    def __count_trie_nodes(self, trie_node) -> int:
        """Iterative (stack-based) node count. Must NOT be recursive: in
        --boundaries_discovery mode the trie can have branches as deep as
        the longest un-anchored sequence in the corpus, which routinely
        exceeds Python's default recursion limit (RecursionError)."""
        if not isinstance(trie_node, dict):
            return 0
        count = 0
        stack = [trie_node]
        while stack:
            node = stack.pop()
            count += 1
            for key, value in node.items():
                if key not in ('IDX', '##') and isinstance(value, dict):
                    stack.append(value)
        return count

    def prepare_encoding(self) -> None:
        self.ids.append(0)
        self.tokens.append("")

    # =========================================================================
    # Suffix-stem pruning
    # =========================================================================

    def __prune_weak_suffixes(self) -> None:
        """Two-pass ++ trie pruning (see v1.4.1 docstring for details)."""
        if self.min_suffix_stems <= 0 or '++' not in self.roots:
            return

        pruned = 0

        def _walk_prune(node: dict, path: str) -> None:
            nonlocal pruned
            for k in list(node.keys()):
                if k in ('IDX', '##'):
                    continue
                if isinstance(node[k], dict):
                    _walk_prune(node[k], path + k)
            if 'IDX' in node:
                n_stems = len(self.suffix_stems.get(path, set()))
                if n_stems < self.min_suffix_stems:
                    del node['IDX']
                    pruned += 1

        _walk_prune(self.roots['++'], '')

        def _has_idx(node: dict) -> bool:
            if 'IDX' in node:
                return True
            return any(
                _has_idx(v) for k, v in node.items()
                if isinstance(v, dict) and k not in ('IDX', '##')
            )

        for k in list(self.roots['++'].keys()):
            if k in ('IDX', '##'):
                continue
            if isinstance(self.roots['++'][k], dict) and not _has_idx(self.roots['++'][k]):
                del self.roots['++'][k]

        if pruned:
            print(f"  Pruned {pruned} ++ suffix(es) with < {self.min_suffix_stems} distinct stems.")

    # =========================================================================
    # Training
    # =========================================================================

    # =========================================================================
    # v1.4.5 — boundary discovery (whitespace-free, utterance-anchored)
    # =========================================================================
    def _anchor_sequences(self, text: str):
        """Cut `text` only at UNAMBIGUOUS edges: punctuation + line/tab breaks
        (each is a true end "]]" of one sequence and a true start "[[" of the
        next).  Registered special tokens (e.g. CHILDES "*CHI:") are treated as
        true starts.  Inside a sequence, spaces are kept as the soft-cue symbol
        SPACE_MARK rather than used as delimiters, so word boundaries must be
        re-discovered from statistics (this is what makes the method work
        identically for spaced eng/nld and unspaced zho)."""
        pieces = []
        split_re = self._special_token_splitter()
        chunks = split_re.split(text) if split_re else [text]
        for chunk in chunks:
            if chunk in self.roots['[RSX]']:
                continue                                   # special = anchor only
            for seg in self._anchor_re.split(chunk):
                s = ''.join(ch for ch in seg
                            if ch.isalpha() or ch in (' ', "'", '-'))
                s = re.sub(r'\s+', ' ', s).strip().replace(' ', self.SPACE_MARK)
                if len(s) >= 2:
                    pieces.append(s)
        return pieces

    def __discover_boundaries(self, seq: str) -> None:
        """Single left-to-right pass over ONE true-anchored sequence.  At each
        position the root trie is read from the true start and the infl trie
        from the true end; a morpheme break "||" is committed as soon as BOTH
        respect the Tolerance Principle (sufficiency).  Pre-req: `seq` has
        already been built into self.roots (forward) and self.infls (reversed).
        Mirrors the bookkeeping of __morsplit so emitted units are retrievable."""
        n = len(seq)
        if n < 3:
            return
        # v1.4.5a: O(n) instead of O(n^2).  The split test at position i only
        # needs consecutive node counts on the forward root path (freq(seq[:i-1]),
        # freq(seq[:i])) and on the backward infl path (freq of the reversed
        # suffixes of length n-i-1 and n-i).  Both are obtained with ONE linear
        # walk each, so we never re-slice/re-traverse from the anchors per i.
        rc = [0] * (n + 1)            # rc[k]   = freq(seq[:k])         (root path)
        rnode = [None] * (n + 1)      # rnode[k]= node at seq[:k]
        node = self.roots
        for k in range(1, n + 1):
            node = node[seq[k - 1]]
            rc[k] = node['##']
            rnode[k] = node
        ic = [0] * (n + 1)            # ic[j]   = freq(seq[n-j:][::-1]) (infl path)
        inode = [None] * (n + 1)
        node = self.infls
        rev = seq[::-1]
        for j in range(1, n + 1):
            node = node[rev[j - 1]]
            ic[j] = node['##']
            inode[j] = node

        last = 0
        for i in range(2, n):         # commit "||" as soon as bilateral TP holds
            if not self.__check_tp(rc[i - 1], rc[i]):
                continue
            if not self.__check_tp(ic[n - i - 1], ic[n - i]):
                continue
            morph, suffix = seq[last:i], seq[i:]
            rn = rnode[i]
            if 'IDX' not in rn:
                rn['IDX'] = 1                              # left segment retrievable
            self.__build_trie(suffix, self.roots['++'])
            self.suffix_stems.setdefault(suffix, set()).add(morph)
            if self.ooa:
                key = self.current_world + " " + morph + "||" + suffix
                if key in self.ooa_split:
                    self.ooa_split[key] += 1
                else:
                    self.ooa_data.append([
                        self.current_world, morph + "||" + suffix,
                        rc[i - 1], rc[i], len(rnode[i]),
                        ic[n - i - 1], ic[n - i], len(inode[n - i]),
                        len(self.types), self._ooa_exposure,
                    ])
                    self.ooa_split[key] = 1
            last = i

    def __train_boundaries_discovery(self, all_contents: str, n_files: int,
                                     output_dir: str) -> None:
        """Whitespace-free training pass: anchor -> sort shorter-first -> build
        the dual trie over full sequences -> place boundaries by sufficiency ->
        shared post-processing (identical to the tail of train())."""
        sequences = self._anchor_sequences(all_contents)
        sequences.sort(key=len)                            # developmental curriculum
        self.num_tokens_in_corpus = len(sequences)
        for s in sequences:
            self.num_chars_in_corpus += len(s)
        print(f"MorPiece (--boundaries_discovery): {n_files} files -> "
              f"{len(sequences)} true-anchored sequences "
              f"(whitespace NOT a boundary; shorter-first).")

        token_counter = 0
        next_ooa_at = self.ooa_token_interval
        self._ooa_exposure = 0                 # running sequence position for the OOA event log
        for seq in sequences:
            self.current_world = seq
            self._ooa_exposure += 1
            self.__build_trie(seq, self.roots)             # root trie  (true start)
            self.__build_trie(seq[::-1], self.infls)       # infl trie  (true end)
            self.__discover_boundaries(seq)
            token_counter += self._ooa_increment(seq)
            if self.ooa and token_counter >= next_ooa_at:
                self.__incremental_cleaning(self.roots, self.min_frequency)
                self.__build_vocab_freq()
                ooa_dir = os.path.join(
                    output_dir, f"ooa_vocab_growth")
                os.makedirs(ooa_dir, exist_ok=True)
                self.save_vocab(os.path.join(ooa_dir, f"vocab_{token_counter}.json"))
                print('.', end='', flush=True)
                while next_ooa_at <= token_counter:
                    next_ooa_at += self.ooa_token_interval
        if self.ooa:
            print("")

        # shared post-processing (mirror tail of train())
        self.types = dict(sorted(self.types.items(), key=lambda x: x[1], reverse=True))
        self.__sort_trie_by_freq(self.roots)
        self.num_chars_in_trie = self.__count_trie_nodes(self.roots)
        self.__prune_weak_suffixes()
        self.__optimize(self.roots)
        self.num_chars_in_optimized_trie = self.__count_trie_nodes(self.roots)
        self._add_frequent_chars()
        print(f"MorPiece (--boundaries_discovery) trained: final vocabulary = "
              f"{self.get_vocab_size()} tokens")

    # =========================================================================
    # OOA exposure accounting (--counter_unit)
    # =========================================================================
    @staticmethod
    def _count_syllables(text: str) -> int:
        """Approximate syllable count for OOA exposure accounting.

        Latin script: number of maximal vowel runs (the standard cheap nucleus
        estimate; slightly under-counts hiatus such as IT 'pa-u-ra').  CJK
        ideographs, Kana morae and Hangul syllable blocks count 1 each (~one
        syllable per glyph), keeping the unit meaningful for zho/jpn/kor under
        --boundaries_discovery.  This is for *setting an interval*, not for
        linguistic analysis: the syllable is used because it is the most
        cross-linguistically comparable measure of input quantity
        (Raesaenen et al. 2019, 2021)."""
        VOWELS = set("aeiouy\u00e0\u00e1\u00e2\u00e3\u00e4\u00e5\u00e8\u00e9"
                     "\u00ea\u00eb\u00ec\u00ed\u00ee\u00ef\u00f2\u00f3\u00f4"
                     "\u00f5\u00f6\u00f9\u00fa\u00fb\u00fc\u00fd\u00ff")
        syl, in_vowel, has_letter = 0, False, False
        for ch in text.lower():
            cp = ord(ch)
            if (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or
                    0xF900 <= cp <= 0xFAFF or 0x3040 <= cp <= 0x30FF or
                    0xAC00 <= cp <= 0xD7A3):
                syl += 1
                in_vowel = False
                has_letter = True
                continue
            if ch in VOWELS:
                has_letter = True
                if not in_vowel:
                    syl += 1
                    in_vowel = True
            else:
                if ch.isalpha():
                    has_letter = True
                in_vowel = False
        if has_letter and syl == 0:        # consonant-only alphabetic token
            return 1
        return syl

    def _ooa_increment(self, text: str) -> int:
        """How much one processed item advances the token-based OOA exposure
        counter, per self.counter_unit: 'tokens'->1, 'chars'->len, 'syllables'->~syl."""
        if self.counter_unit == 'chars':
            return len(text)
        if self.counter_unit == 'syllables':
            return self._count_syllables(text)
        return 1

    def train(self, training_corpus_path: str, text_column: str = "text",
              save_complete_tries=None, output_dir: str = "tokenizer") -> None:
        """
        Train the MorPiece tokenizer.

        Parameters
        ----------
        training_corpus_path : str
            Path to a directory of UTF-8 text files OR a single .parquet file.
        text_column : str
            Column name in the parquet file (default 'text').
        save_complete_tries : str | bool | None
            If set, dump the COMPLETE, un-pruned root + inflection tries to a
            JSON file *before* any min_frequency / min_suffix_stems pruning is
            applied (see "Changes in 1.4.3").  Pass a path string, or `True`
            for the default 'tokenizer/complete_tries.json'.  Default None
            (disabled) — training is unaffected when the option is not used.
            The resulting file is consumed by morpiece_trie_explorer.py.
        output_dir : str
            Root directory for OOA snapshots (default 'tokenizer').  Both
            type_based and token-based snapshots land in the same
            'ooa_cutoff{c}_mfreq{m}' subfolder; the filename (vocab_type_* vs
            vocab_*) carries the mode distinction.
        """
        all_contents = ""
        n_files = 0

        training_corpus_path = str(training_corpus_path)
        if training_corpus_path.endswith(".parquet"):
            try:
                import pandas as pd
            except ImportError:
                raise ImportError("pandas required: pip install pandas pyarrow")
            df = pd.read_parquet(training_corpus_path)
            if text_column not in df.columns:
                raise ValueError(
                    f"Column '{text_column}' not found. Available: {df.columns.tolist()}"
                )
            for content in df[text_column].dropna().astype(str):
                content = self._preprocess_text(content)
                all_contents += content + "\n"
                n_files += 1
            print(
                f"MorPiece tokenizer training: loaded {n_files} rows from "
                f"parquet '{training_corpus_path}' (column='{text_column}')..."
            )
        else:
            for filename in os.listdir(training_corpus_path):
                file_path = os.path.join(training_corpus_path, filename)
                if os.path.isfile(file_path):
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    content = self._preprocess_text(content)
                    all_contents += content + "\n"
                    n_files += 1

        # v1.4.5: boundary-discovery mode never tokenises on whitespace.
        if self.boundaries_discovery:
            self.__train_boundaries_discovery(all_contents, n_files, output_dir)
            return

        words = all_contents.split()
        print(
            f"MorPiece tokenizer training: processing corpus "
            f"({n_files} files, {len(words)} tokens)..."
        )

        if self.ooa:
            if self.type_based:
                print(
                    f"Saving Order Of Acquisition Vocabulary every "
                    f"{self.ooa_type_interval} types (frequency-ordered) (.):"
                )
            else:
                print(f"Saving Order Of Acquisition Vocabulary every "
                      f"{self.ooa_token_interval} {self.counter_unit} (.):")

        # --- type-frequency counting pass ------------------------------------
        for word in words:
            # v1.4.4: a registered special token (e.g. CHILDES "*CHI:") is
            # atomic — count it as a corpus token but keep it out of `types`
            # so it neither competes for vocab nor enters the morpheme trie.
            if word in self.roots['[RSX]']:
                self.num_tokens_in_corpus += 1
                self.num_chars_in_corpus  += len(word)
                continue
            word_alpha = ''.join(
                ch for ch in word if ch.isalpha() or ch in ("'", "-")
            )
            word = word_alpha if word_alpha else word
            if word:
                self.types[word] = self.types.get(word, 0) + 1
                self.num_tokens_in_corpus += 1
                self.num_chars_in_corpus  += len(word)

        # --- choose training order -------------------------------------------
        if self.type_based:
            if self.ooa:
                training_items = sorted(
                    self.types.keys(), key=lambda w: self.types[w], reverse=True
                )
                print(
                    f"  type_based=True (ooa): iterating {len(training_items)} types "
                    f"in descending frequency order."
                )
            else:
                training_items = list(self.types.keys())
            print(
                f"  type_based=True: {len(training_items)} unique types "
                f"(corpus {self.num_tokens_in_corpus} tokens, "
                f"TTR={round(len(self.types)/self.num_tokens_in_corpus, 3)})"
            )
        else:
            training_items = words

        # --- main trie-building pass -----------------------------------------
        token_counter = 0
        next_ooa_at = self.ooa_token_interval
        self._ooa_exposure = 0                 # running token position for the OOA event log
        for word in training_items:
            # v1.4.4: registered special tokens stay atomic — they already
            # live in [RSX] with their own IDX, so do not morsplit them.
            if word in self.roots['[RSX]']:
                self.current_world = word
                self._ooa_exposure += 1
                token_counter += 1 if self.type_based else self._ooa_increment(word)
                continue
            word_alpha = ''.join(
                ch for ch in word if ch.isalpha() or ch in ("'", "-")
            )
            word = word_alpha if word_alpha else word
            if not word:
                continue

            self.current_world = word
            self._ooa_exposure += 1            # stamp this token's position before morsplit
            self.__build_trie(word[::-1], self.infls)
            self.__build_trie(word, self.roots)
            self.__morsplit(word)
            token_counter += 1 if self.type_based else self._ooa_increment(word)

            # -----------------------------------------------------------------
            # OOA snapshots
            #   type_based : every `ooa_type_interval` TYPES (exposure = types).
            #   token-based: every `ooa_token_interval` units of `counter_unit`
            #                (tokens | chars | syllables).  Threshold-crossing
            #                (>=) instead of modulo, so chars/syllables — whose
            #                per-item increment exceeds 1 — never skip a snapshot.
            # -----------------------------------------------------------------
            if self.ooa:
                ooa_dir = os.path.join(
                    output_dir, f"ooa_vocab_growth"
                )
                if self.type_based:
                    if token_counter % self.ooa_type_interval == 0:
                        snap = self.__build_vocab_freq_snapshot()
                        os.makedirs(ooa_dir, exist_ok=True)
                        snap_path = os.path.join(ooa_dir, f"vocab_type_{token_counter}.json")
                        with open(snap_path, 'w') as f:
                            json.dump({'vocab': snap}, f, indent=2)
                        print('.', end='', flush=True)
                elif token_counter >= next_ooa_at:
                    self.__incremental_cleaning(self.roots, self.min_frequency)
                    self.__build_vocab_freq()
                    os.makedirs(ooa_dir, exist_ok=True)
                    self.save_vocab(os.path.join(ooa_dir, f"vocab_{token_counter}.json"))
                    print('.', end='', flush=True)
                    while next_ooa_at <= token_counter:
                        next_ooa_at += self.ooa_token_interval

        if self.ooa:
            print("")

        self.types = dict(sorted(self.types.items(), key=lambda x: x[1], reverse=True))
        self.__sort_trie_by_freq(self.roots)
        self.num_chars_in_trie = self.__count_trie_nodes(self.roots)

        # --- v1.4.3: optional snapshot of the COMPLETE (un-pruned) tries -----
        # Taken here, after frequency-sorting but BEFORE __prune_weak_suffixes()
        # and BEFORE __optimize() — i.e. before any min_suffix_stems / min_freq
        # pruning.  Every node still carries its raw '##' frequency count.
        if save_complete_tries:
            ct_path = (save_complete_tries if isinstance(save_complete_tries, str)
                       else os.path.join(output_dir, "complete_tries.json"))
            self.save_complete_tries(ct_path)

        self.__prune_weak_suffixes()

        print(
            f"MorPiece tokenizer training: trie optimisation "
            f"(vocab_size={self.vocab_size}, min_freq={self.min_frequency}, "
            f"cutoff={self.cutoff}, "
            f"min_suffix_stems={self.min_suffix_stems})..."
        )
        self.__optimize(self.roots)
        self.num_chars_in_optimized_trie = self.__count_trie_nodes(self.roots)

        self._add_frequent_chars()

        print(f"MorPiece tokenizer trained: final vocabulary = {self.get_vocab_size()} tokens")
        if self.ooa:
            print("MorPiece tokenizer Order of Acquisition data prepared.")

    # =========================================================================
    # Encoding / Decoding
    # =========================================================================

    def encode(self, sentence: str):
        """Encode a sentence into (token_ids, token_strings)."""
        sentence = self._preprocess_text(sentence)
        self.ids, self.tokens = [], []
        for word in sentence.strip().split():
            if word in self.roots['[RSX]']:
                self.ids.append(self.roots['[RSX]'][word]['IDX'])
                self.tokens.append(word)
            else:
                self.prepare_encoding()
                self.__retrieve(word, self.roots)
        if self.use_tokenizers_lib:
            # v1.4.6 fix: _postprocess_tokens prepends BOS to the ID list only.
            # Mirror that prepend in the token list so the two returned lists
            # stay index-aligned — callers (calculate_stats, diagnose_tp, the
            # multilingual router, anything that zips ids with tokens) relied on
            # len(ids) == len(tokens), which silently broke after the BOS prepend.
            n_before = len(self.ids)
            self.ids = self._postprocess_tokens(self.ids)
            if len(self.ids) == n_before + 1:
                self.tokens = [self.start_of_text_symbol] + self.tokens
        return self.ids, self.tokens

    def decode(self, sentence_idxs: list) -> list:
        tokens = [self.id_to_vocab.get(idx, '<unk>') for idx in sentence_idxs]
        if self.use_tokenizers_lib and tokens:
            text = ''.join(tokens)
            try:
                return self.decoder.decode(text).split()
            except Exception:
                return tokens
        return tokens

    # =========================================================================
    # Special-token management
    # =========================================================================

    def set_special_tokens(self, token_list: list) -> None:
        for item in token_list:
            if item not in self.roots['[RSX]']:
                self.roots['[RSX]'][item] = {'IDX': self.idx}
                self.idx += 1

    def pad_sentence(self, sentence: str, l: int, pad: str = '<pad>') -> str:
        words = sentence.split()
        n_pad = max(l - len(words), 0)
        return ' '.join([pad] * n_pad + words)

    # =========================================================================
    # Diagnostics
    # =========================================================================

    def diagnose_tp(self, sample_words: list = None, n_words: int = 20) -> None:
        """
        Print a per-word trace of why TP does or does not fire, to help
        calibrate cutoff / type_based / min_suffix_stems parameters.

        Uses the suffix_stems dictionary (populated during training) together
        with the vocabulary to report whether splits were found and how many
        stems each suffix derives from.  Also calls encode() on each sample
        word to show the final tokenization.

        Parameters
        ----------
        sample_words : list of str, optional
            Words to trace.  If None, the 20 most-frequent types are used.
        n_words : int
            How many words to show when sample_words is None (default 20).

        Example
        -------
        >>> mop.diagnose_tp(['cantava', 'telefono', 'asociale', 'asola'])
        """
        if not self.types:
            print("No type counts available — train first.")
            return

        if sample_words is None:
            sample_words = [w for w, _ in
                            sorted(self.types.items(), key=lambda x: -x[1])[:n_words]]

        print(f"diagnose_tp  cutoff={self.cutoff}  type_based={self.type_based}  "
              f"min_suffix_stems={self.min_suffix_stems}\n")

        stem_suffix_pairs = set()
        for suf, stems in self.suffix_stems.items():
            for s in stems:
                stem_suffix_pairs.add((s, suf))

        for w in sample_words:
            ids_out, toks_out = self.encode(w)
            in_vocab = (w in (self.vocab_to_id or {})) or (w in self.roots.get('[RSX]', {}))
            freq      = self.types.get(w, 0)

            print(f"  Word: {repr(w)}  (corpus freq={freq})")
            print(f"    encode → {toks_out}")
            print(f"    in vocabulary: {in_vocab}")

            found_splits = [(s, suf) for s, suf in stem_suffix_pairs
                            if s + suf == w and len(s) >= 1 and len(suf) >= 1]
            if found_splits:
                print(f"    Splits proposed during training:")
                for stem, suf in sorted(found_splits, key=lambda x: len(x[0])):
                    n_stems = len(self.suffix_stems.get(suf, set()))
                    survives = n_stems >= self.min_suffix_stems
                    status   = (f"++ kept  ({n_stems} stems ≥ {self.min_suffix_stems})"
                                if survives else
                                f"++ PRUNED ({n_stems} stems < {self.min_suffix_stems})")
                    print(f"      {repr(stem)} | {repr(suf)}  → {status}")
            else:
                reasons = []
                if freq == 0:
                    reasons.append("word never seen in corpus")
                else:
                    reasons.append("no split fired during training")
                    reasons.append(f"possible causes: cutoff too high (need freq>{self.cutoff}); "
                                   f"word too short or non-decomposable; "
                                   f"infl-trie ending below cutoff")
                print(f"    No splits proposed: {'; '.join(reasons)}")
            print()

        print(f"  === Global summary ===")
        print(f"  Total ++ suffixes proposed: {len(self.suffix_stems)}")
        top = sorted(self.suffix_stems.items(), key=lambda x: -len(x[1]))[:10]
        for suf, stems in top:
            survives = len(stems) >= self.min_suffix_stems
            mark     = "✓" if survives else "✗"
            print(f"  {mark} {repr(suf):12s} {len(stems):3d} stems  e.g. {sorted(list(stems))[:3]}")
        n_pp = len([t for t in (self.vocab_to_id or {}) if t.startswith('++')])
        print(f"  ++ tokens in final vocabulary: {n_pp}")
        print(f"  Parameter advice:")
        if n_pp == 0 and len(self.suffix_stems) == 0:
            print(f"    No splits fired at all.")
            print(f"    → Lower cutoff (current={self.cutoff}) if corpus is small.")
            print(f"    → Ensure type_based=False for token-frequency-driven TP.")
        elif n_pp == 0:
            print(f"    Splits proposed but all pruned by min_suffix_stems={self.min_suffix_stems}.")
            print(f"    → Lower min_suffix_stems or expand corpus for more stem diversity.")
        else:
            print(f"    ++ tokens present — tokenizer is segmenting. Tune to taste.")

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_num_chars_in_trie(self):           return self.num_chars_in_trie
    def get_num_chars_in_optimized_trie(self): return self.num_chars_in_optimized_trie
    def get_num_chars_in_corpus(self):         return self.num_chars_in_corpus
    def get_num_tokens_in_corpus(self):        return self.num_tokens_in_corpus
    def get_num_types_in_corpus(self):         return len(self.types)
    def get_compression_ratio(self):           return round(self.num_chars_in_optimized_trie / self.num_chars_in_trie * 100, 3)
    def get_ttr(self):                         return round(len(self.types) / self.num_tokens_in_corpus, 3)

    def get_vocab_size(self) -> int:
        self.vocab_size = self.idx
        return self.idx

    # =========================================================================
    # Serialisation
    # =========================================================================

    def from_pretrained(self, load_file: str) -> None:
        tokenizer_path = os.path.join(load_file, 'tokenizer.json')
        print(f"Loading MorPiece tokenizer from {tokenizer_path}...")
        with open(tokenizer_path, 'r', encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and 'roots' in data:
            self.roots       = data['roots']
            self.vocab_to_id = data.get('vocab', {})
            self.id_to_vocab = {v: k for k, v in self.vocab_to_id.items()}
            if self.vocab_to_id:
                self.idx = max(self.vocab_to_id.values()) + 1
            if 'special_token_ids' in data:
                sp = data['special_token_ids']
                self.unk_token_id  = sp.get('unk',  self.unk_token_id)
                self.pad_token_id  = sp.get('pad',  self.pad_token_id)
                self.bos_token_id  = sp.get('bos',  self.bos_token_id)
                self.eos_token_id  = sp.get('eos',  self.eos_token_id)
                self.mask_token_id = sp.get('mask', self.mask_token_id)
        else:
            self.roots       = data
            self.vocab_to_id = {}
            self.id_to_vocab = {}
        if '[RSX]' not in self.roots:
            raise ValueError("Invalid tokenizer format: missing [RSX] root node.")
        # A loaded tokenizer may carry special tokens that the splitter cache
        # has not seen — force a rebuild on next _preprocess_text call.
        self._split_re_n = -1

    def save_pretrained(self, save_file: str) -> None:
        self.__build_vocab_lookup()
        with open(save_file, 'w', encoding='utf-8') as f:
            json.dump({
                'roots': self.roots,
                'vocab': self.vocab_to_id,
                'special_token_ids': {
                    'unk': self.unk_token_id, 'pad': self.pad_token_id,
                    'bos': self.bos_token_id, 'eos': self.eos_token_id,
                    'mask': self.mask_token_id,
                },
            }, f, indent=2)

    def save_complete_tries(self, save_file: str) -> None:
        """
        Dump the COMPLETE, UN-PRUNED tries to a single JSON file (v1.4.3).

        Called from train() before __prune_weak_suffixes() and __optimize()
        when the save_complete_tries option is requested.  The dump preserves:
          • every node's raw '##' frequency count;
          • the placeholder 'IDX' markers (a registered word/morpheme path);
          • the reversed-word inflection trie (self.infls);
          • the ++ suffix→stems map backing min_suffix_stems pruning.

        Together these let a downstream tool reconstruct exactly which
        branches the later min_frequency cut will remove and what frequency
        each surviving / discarded node carried.  Consumed by
        morpiece_trie_explorer.py.

        NOTE: in token-based mode (type_based=False) with ooa=True the trie
        has already been thinned every 100 000 tokens by
        __incremental_cleaning, so 'complete' there means 'complete as of
        end-of-training', not 'every node ever created'.
        """
        out_dir = os.path.dirname(save_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        payload = {
            "format": "morpiece-complete-tries/1",
            "meta": {
                "version": __version__,
                "vocab_size": self.vocab_size,
                "min_frequency": self.min_frequency,
                "cutoff": self.cutoff,
                "min_suffix_stems": self.min_suffix_stems,
                "type_based": self.type_based,
                "num_tokens_in_corpus": self.num_tokens_in_corpus,
                "num_types_in_corpus": len(self.types),
            },
            "roots": self.roots,
            "infls": self.infls,
            "suffix_stems": {suf: sorted(stems)
                             for suf, stems in self.suffix_stems.items()},
        }
        with open(save_file, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"MorPiece complete (un-pruned) tries saved → {save_file}")
        print(f"  root nodes={self.__count_trie_nodes(self.roots)}  "
              f"infl nodes={self.__count_trie_nodes(self.infls)}  "
              f"++ suffixes={len(self.suffix_stems)}")

    def save_vocab(self, save_file: str) -> None:
        self.__sort_trie_by_freq(self.roots)
        self.__build_vocab_lookup()
        with open(save_file, 'w') as f:
            json.dump({'vocab': self.vocab_to_freq}, f, indent=2)

    def save_types(self, file: str) -> None:
        with open(file, 'w') as f:
            json.dump(self.types, f, indent=2)

    def save_ooa(self, file: str) -> None:
        with open(file, 'w', encoding='utf8') as f:
            for row in self.ooa_data:
                f.write('\t'.join(str(cell) for cell in row) + '\n')

    def save_HF(self, save_directory: str, model_max_length: int = 1024) -> None:
        """Save in HuggingFace PreTrainedTokenizerFast format."""
        os.makedirs(save_directory, exist_ok=True)
        if not hasattr(self, 'vocab_to_id') or self.vocab_to_id is None:
            self.__build_vocab_lookup()

        vocab = [""] * len(self.vocab_to_id)
        for token, token_id in self.vocab_to_id.items():
            if token_id < len(vocab):
                vocab[token_id] = token
        for i in range(len(vocab)):
            if vocab[i] == "":
                vocab[i] = "<unk>"

        sp = {
            "<unk>": self.unk_token_id, "<pad>": self.pad_token_id,
            "<s>":   self.bos_token_id, "</s>":  self.eos_token_id,
            "<mask>": self.mask_token_id,
        }
        for token, token_id in sp.items():
            if token_id < len(vocab):
                vocab[token_id] = token

        tokenizer_json = {
            "version": "1.0", "truncation": None, "padding": None,
            "added_tokens": [
                {"id": sp[tok], "content": tok, "single_word": False,
                 "lstrip": False, "rstrip": False, "normalized": False, "special": True}
                for tok in ("<unk>", "<pad>", "<s>", "</s>", "<mask>")
            ],
            "normalizer": {
                "type": "Sequence",
                "normalizers": [{"type": "Lowercase"}, {"type": "NFKC"}],
            },
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": {
                "type": "TemplateProcessing",
                "single": [
                    {"SpecialToken": {"id": "<s>", "type_id": 0}},
                    {"Sequence":     {"id": "A",   "type_id": 0}},
                ],
                "pair": [
                    {"SpecialToken": {"id": "<s>", "type_id": 0}},
                    {"Sequence":     {"id": "A",   "type_id": 0}},
                    {"SpecialToken": {"id": "<s>", "type_id": 1}},
                    {"Sequence":     {"id": "B",   "type_id": 1}},
                ],
                "special_tokens": {
                    "<s>":  {"id": "<s>",  "ids": [sp["<s>"]],  "tokens": ["<s>"]},
                    "</s>": {"id": "</s>", "ids": [sp["</s>"]], "tokens": ["</s>"]},
                },
            },
            "decoder": {"type": "WordPiece", "prefix": "++", "cleanup": True},
            "model": {
                "type": "WordPiece", "unk_token": "<unk>",
                "continuing_subword_prefix": "++",
                "max_input_chars_per_word":  100,
                "vocab": {token: i for i, token in enumerate(vocab) if token.strip()},
            },
        }
        with open(os.path.join(save_directory, "tokenizer.json"), 'w', encoding='utf-8') as f:
            json.dump(tokenizer_json, f, indent=2, ensure_ascii=False)

        added_tokens_decoder = {
            str(tid): {"content": tok, "lstrip": False, "normalized": False,
                       "rstrip": False, "single_word": False, "special": True}
            for tok, tid in sp.items()
        }
        tokenizer_config = {
            "added_tokens_decoder": added_tokens_decoder,
            "bos_token": "<s>", "clean_up_tokenization_spaces": False,
            "eos_token": "</s>", "extra_special_tokens": {},
            "mask_token": "<mask>", "model_max_length": model_max_length,
            "pad_token": "<pad>", "tokenizer_class": "PreTrainedTokenizerFast",
            "unk_token": "<unk>",
        }
        with open(os.path.join(save_directory, "tokenizer_config.json"), 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)

        special_tokens_map = {
            "bos_token": "<s>", "eos_token": "</s>",
            "unk_token": "<unk>", "pad_token": "<pad>", "mask_token": "<mask>",
        }
        with open(os.path.join(save_directory, "special_tokens_map.json"), 'w', encoding='utf-8') as f:
            json.dump(special_tokens_map, f, indent=2, ensure_ascii=False)

        with open(os.path.join(save_directory, "vocab.txt"), 'w', encoding='utf-8') as f:
            for token in vocab:
                if token.strip():
                    f.write(token + "\n")

        print(f"MorPiece tokenizer saved in HuggingFace format → {save_directory}")
        print(f"  vocab_size={len([t for t in vocab if t.strip()])}  model_max_length={model_max_length}")
        print(f"  Load: PreTrainedTokenizerFast.from_pretrained('{save_directory}')")

    # =========================================================================
    # HF Tokenizer factory helpers
    # =========================================================================

    def create_WordPiece_tokenizer(self):
        if not self.use_tokenizers_lib:
            print("Tokenizers library integration is disabled.")
            return None
        from tokenizers import Tokenizer
        from tokenizers.models import WordPiece
        if not hasattr(self, 'vocab_to_id') or self.vocab_to_id is None:
            self.__build_vocab_lookup()
        vocab = [""] * len(self.vocab_to_id)
        for token, token_id in self.vocab_to_id.items():
            if token_id < len(vocab):
                vocab[token_id] = token
        for i in range(len(vocab)):
            if vocab[i] == "":
                vocab[i] = "<unk>"
        tokenizer = Tokenizer(WordPiece({t: i for i, t in enumerate(vocab)}, unk_token="<unk>"))
        tokenizer.normalizer     = self.normalizer
        tokenizer.pre_tokenizer  = self.pre_tokenizer
        tokenizer.decoder        = self.decoder
        tokenizer.post_processor = self.post_processor
        tokenizer.save()
        return tokenizer

    def create_bpe_tokenizer(self):
        if not self.use_tokenizers_lib:
            print("Tokenizers library integration is disabled.")
            return None
        try:
            from tokenizers import Tokenizer
            from tokenizers.models import BPE
        except ImportError:
            print("tokenizers library not available.")
            return None
        if not hasattr(self, 'vocab_to_id') or self.vocab_to_id is None:
            self.__build_vocab_lookup()
        if not self.vocab_to_id:
            print("No vocabulary found. Train the tokenizer first.")
            return None
        vocab    = [""] * len(self.vocab_to_id)
        for token, token_id in self.vocab_to_id.items():
            if token_id < len(vocab):
                vocab[token_id] = token
        for i in range(len(vocab)):
            if vocab[i] == "":
                vocab[i] = "<unk>"
        vocab_set = set(vocab)
        merges    = []

        def extract_merges(trie, path=""):
            if not isinstance(trie, dict):
                return
            for key, subtrie in trie.items():
                if key in ('IDX', '##') or key in self.reserved_keys:
                    continue
                current_path = path + key
                if current_path in vocab_set and isinstance(subtrie, dict):
                    for child_key in subtrie:
                        if child_key not in ('IDX', '##') and child_key not in self.reserved_keys:
                            child_path = current_path + child_key
                            if child_path in vocab_set:
                                pair = (current_path, child_key)
                                if pair not in merges:
                                    merges.append(pair)
                extract_merges(subtrie, current_path)

        for root_key, root_trie in self.roots.items():
            if root_key != '[RSX]':
                extract_merges(root_trie)

        vocab_dict = {token: i for i, token in enumerate(vocab)}
        bpe_merges = [" ".join(m) for m in merges[:1000]]
        try:
            tokenizer = Tokenizer(BPE(vocab=vocab_dict, merges=bpe_merges, unk_token="<unk>"))
            tokenizer.normalizer     = self.normalizer
            tokenizer.pre_tokenizer  = self.pre_tokenizer
            tokenizer.decoder        = self.decoder
            tokenizer.post_processor = self.post_processor
            print(f"BPE tokenizer: {len(vocab_dict)} vocab items, {len(bpe_merges)} merges")
            return tokenizer
        except Exception as e:
            print(f"BPE creation failed: {e}")
            return None