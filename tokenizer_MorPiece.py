__version__ = "1.3.1"
__author__ = "Cristiano Chesi & NeTS Lab @ IUSS (with Claude Sonnet 4.6 fixes and optimizations)"
__email__ = "cristiano.chesi@iusspavia.it"
__status__ = "Research"
__date__ = "2026-05-15"
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

If the "order or acquisition" (ooa) parameter is set to True, each 100K of exposure a vocabulary is created to check splitting hypotheses postulated and the evidence needed (for research purposes)

+ version 1.2.* supports HF tokenizers pre and post processing standard routines

Examples:

    import tokenizer_MorPiece as MoP

    mop = MoP.MorPiece(vocab_size=vocab_size, cutoff=cutoff, min_frequency=min_frequency, bf=bf, ooa=ooa, use_tokenizers_lib=True)
    mop.train(text)
    mop.save('./mop_tokenizer/tokenizer.json')

    s = "test sentence"
    print("Sentence to tokenize: " + s)
    ids, tokens = mp.encode(s)
    print(ids, tokens)

Todo:

    - Evaluate different splitting algorithms (Tolerance Principle / Sufficiency Principle in the root-trie only)
    - Multi word evaluation before splitting

Reference:

    https://github.com/cristianochesi/morpiece
"""

import os
import json
from collections import deque
from math import log
from tokenizers import pre_tokenizers, decoders, normalizers, Regex, processors


class MorPiece:
    """MorPiece incrementally chunks words into potentially meaningful morphemes.
    The training consider an incremental word-by-word traversal and build a "root" and an "infl" trie, consisting of any token found at least once in the corpus.
    Token (or Type, when type_based argument is set) frequency is indicated at each node in the trie.
    The splitting procedure consists of evaluating if the Tolerance Principle (Yang 2016) applies at every character every time an incoming word "traverses" the lexicon.

    Attributes
    ----------
    vocab_size : int
        the maximum number of items in the vocabulary (default 30000)
    min_frequency : int
        the minimum frequency of a token to be included in the vocabulary (default 2)
    cutoff : int
        the minimum frequency considered for the mother node before the sufficiency principle is applied (default 8)
    bf : int
        the minimum number of daughters nodes that must be present (i.e., branching factor in the trie) before applying the sufficiency principle (default 10)
    special_tokens : list
        a list of special tokens to be reserved (in None is passed, the default ones are '<unk>', '<pad>', '<s>', '</s>')
    ooa : boolean
        plot the Order of Acquisition of each split (default is False)
    use_tokenizers_lib : boolean
        plot the Order of Acquisition of each split (default is False)
    type_based : boolean
        use types instead of tokens to calculate TP

    Methods
    -------
    train(corpus=None)
        train the tokenizer using a text file (UTF-8 encoding preferred)
    """

    ids: list[int]
    tokens: list[str]
    vocab_size: int

    def __init__(self, vocab_size=30000, min_frequency=10, cutoff=100, bf=10, special_tokens=None, ooa=True, use_tokenizers_lib=False, type_based=False):
        """
        Parameters
        ----------
        vocab_size : int
            the maximum number of items in the vocabulary (default 30000)
        min_frequency : int
            the minimum frequency of a token to be included in the vocabulary (default 2)
        cutoff : int
            the minimum frequency considered for the mother node before the sufficiency principle is applied (default 8)
        bf : int
            the minimum number of daughters nodes that must be present (i.e., branching factor in the trie) before applying the sufficiency principle (default 10)
        special_tokens : dict
        use_tokenizers_lib : bool
            whether to use tokenizers library for normalization and pre-processing (default True)
        """
        if special_tokens is None:
            special_tokens = ['<unk>', '<pad>', '<s>', '</s>', '<mask>', '<sep>', '<cls>']
        self.special_tokens = special_tokens
        self.unk_token_id = 0
        self.pad_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        self.mask_token_id = 4
        self.sep_token_id = 5
        self.cls_token_id = 6
        self.start_of_text_symbol = '<s>'
        self.reserved_keys = {'[RSX]', '##', 'IDX', '++'}
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.bf = bf
        self.ooa = ooa
        self.ooa_split = {}
        self.use_tokenizers_lib = use_tokenizers_lib
        self.type_based = type_based
        self.roots = {'[RSX]': {}, '++': {}}
        self.infls = {}
        self.types = {}
        self.last_item_in_trie = {}
        self.idx = 0
        self.ids = []
        self.tokens = []
        self.cutoff = cutoff  # ln(8) is > 2, so, non-branching paths will be ignored
        self.num_tokens_in_corpus = 0
        self.num_chars_in_corpus = 0
        self.num_chars_in_trie = 0
        self.num_chars_in_optimized_trie = 0
        self.set_special_tokens(self.special_tokens)
        self.ooa_data = [['Word', 'Piece', 'Root Trie - Mother freq', 'Root Trie - Daughter freq',
                          'Root Trie - Mother bf (# of daughters)', 'Infl Trie - Mother freq',
                          'Infl Trie - Daughter freq', 'Infl Trie - Mother bf (# of daughters)' 'Vocabulary size',
                          'Token exposure']]
        self.vocab_to_id = None
        self.vocab_to_freq = None
        self.id_to_vocab = None
        self.current_world = ""
        self.node = {}
        self.m_freq = 1
        self.d_freq = 1
        self.m_daughters = 1

        # Initialize tokenizers components if enabled
        if self.use_tokenizers_lib:
            self._init_tokenizers_components()

    def _init_tokenizers_components(self):
        """Initialize the tokenizers library components"""

        # Set up normalizer
        self.normalizer = normalizers.Sequence([
            normalizers.Lowercase(),  # Force lowercase
            normalizers.Prepend(" "),
            normalizers.NFKC(),
            normalizers.Replace(Regex("\n"), '\n '),
            normalizers.Replace(Regex(" *\n"), '\n'),
        ])

        # Set up pre_tokenizer
        self.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(
                Regex(
                    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*| ?\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"),
                behavior="isolated",
                invert=False
            ),
            pre_tokenizers.Split(
                Regex(".{1,24}"),
                behavior="isolated",
                invert=False
            )
        ])

        # Set up decoder
        self.decoder = decoders.Sequence([
            decoders.Strip(' ', 1, 0),
            decoders.Replace("\n ", "\n")
        ])

        # Set up post_processor
        self.post_processor = processors.TemplateProcessing(
            single=f"{self.start_of_text_symbol} $A",
            pair=f"{self.start_of_text_symbol} $A {self.start_of_text_symbol} $B",
            special_tokens=[
                (self.start_of_text_symbol, self.bos_token_id),
            ]
        )

    def _preprocess_text(self, text):
        """Apply tokenizers library preprocessing if enabled"""
        if not self.use_tokenizers_lib:
            return text

        # Apply normalizer
        normalized = self.normalizer.normalize_str(text)

        # Apply pre_tokenizer to get tokens with offsets
        pre_tokenized = self.pre_tokenizer.pre_tokenize_str(normalized)

        # Extract just the tokens (ignoring offsets for now)
        tokens = [token for token, _ in pre_tokenized]
        return " ".join(tokens)

    def _postprocess_tokens(self, token_ids):
        """Apply post-processing if enabled"""
        if not self.use_tokenizers_lib:
            return token_ids

        # For now, just add BOS token at the beginning if not present
        if token_ids and token_ids[0] != self.bos_token_id:
            return [self.bos_token_id] + token_ids
        return token_ids

    def __morsplit(self, w) -> None:
        reversed_w = w[::-1]
        for i in range(2, len(w) + 1):
            self.__find_path(reversed_w[:len(w) + 2 - i], self.infls)
            i_m, i_d, i_bf = self.m_freq, self.d_freq, self.m_daughters
            infls_tp = self.__check_tp(self.m_freq, self.d_freq, self.m_daughters)

            self.__find_path(w[:i], self.roots)
            r_m, r_d, r_bf = self.m_freq, self.d_freq, self.m_daughters
            roots_tp = self.__check_tp(self.m_freq, self.d_freq, self.m_daughters)

            if roots_tp and infls_tp:
                if self.ooa:
                    if self.current_world+" "+w[:i-1]+"-" + w[i-1:] in self.ooa_split:
                        self.ooa_split[self.current_world+" "+w[:i-1]+"-" + w[i-1:]] += 1
                    else:
                        self.ooa_data.append([self.current_world, w[:i - 1] + "-" + w[i - 1:], r_m, r_d, r_bf, i_m, i_d, i_bf,
                                 len(self.types), self.num_tokens_in_corpus])
                        self.ooa_split[self.current_world + " " + w[:i - 1] + "-" + w[i - 1:]] = 1
                if 'IDX' not in self.node:
                    self.node['IDX'] = 1
                self.__build_trie(w[i - 1:], self.roots['++'])

    def __build_trie(self, wordpiece: str, root: dict) -> None:
        """
        Iterative trie construction — O(len(wordpiece)) stack depth replaced with
        a simple loop, eliminating any risk of hitting Python's recursion limit.
        """
        node = root
        for ch in wordpiece:
            if ch in node:
                node[ch]['##'] += 1
            else:
                node[ch] = {'##': 1}
            node = node[ch]
        # Mark terminal node (end of a valid word/piece)
        if 'IDX' not in node:
            node['IDX'] = 1

    @staticmethod
    def __incremental_cleaning(trie, freq):
        """ Remove pendants with low frequency """
        queue = deque([(None, None, trie)])  # (parent, key_in_parent, current_dict)
        while queue:
            parent, key, current = queue.popleft()
            if isinstance(current, dict):
                if key == "[RSX]":
                    continue
                # Remove node if "##" <= min_frequency
                if (key != "++") and (current.get("##", 0) <= freq):
                    if parent is not None:
                        del parent[key]
                        continue  # Skip further processing
                # Remove node "##" with frequency indication
                for k, v in list(current.items()):
                    if isinstance(v, dict):
                        queue.append((current, k, v))

    def __optimize(self, trie):
        """ Assign idx based on word freq and add potential inflection links in the root trie, remove frequency at the end """
        queue = deque([(None, None, trie)])  # (parent, key_in_parent, current_dict)
        while queue:
            parent, key, current = queue.popleft()
            if isinstance(current, dict):
                if key == "[RSX]":
                    continue
                # Remove node if "##" <= min_frequency
                if (key != "++") and (current.get("##", 0) <= self.min_frequency):
                    if parent is not None:
                        del parent[key]
                        continue  # Skip further processing
                # Remove node "##" with frequency indication
                if current.get("##", None) is not None:
                    del current["##"]
                # Renumber IDX if present
                if "IDX" in current:
                    if self.idx < self.vocab_size:
                        current["IDX"] = self.idx
                        self.idx += 1
                    else:
                        del parent[key]
                # Enqueue children for BFS
                for k, v in list(current.items()):
                    if isinstance(v, dict):
                        queue.append((current, k, v))
        self.__build_vocab_lookup()

    def __find_path(self, word: str, trie: dict) -> None:
        """
        Iterative path traversal — walks character-by-character without recursion.
        Updates self.node / self.m_freq / self.d_freq / self.m_daughters at each step,
        identical semantics to the original recursive version.
        """
        node = trie
        for ch in word:
            self.node        = node[ch]
            self.m_freq      = self.d_freq
            self.d_freq      = node[ch]['##']
            self.m_daughters = len(node[ch])
            node             = node[ch]

    def __retrieve(self, string: str, trie: dict) -> None:
        """
        Iterative max-length trie retrieval — replaces the recursive version.

        For each character in *string*:
          • If found in the current trie node  → extend the current token and
            advance into the child node.
          • If found in the '++' suffix root   → start a new '++' token and
            advance into the matching child.
          • Otherwise                          → emit <unk> and stop
            (identical to the original: the remaining characters are dropped).
        """
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
                break   # original behaviour: stop on unknown character

    def __check_tp(self, m, d,
                   nd):  # verify if Tolerance Principle applies between m(other) and d(aughter) nodes, nd indicates the number of daughters
        if not ((m > self.cutoff) and (nd < self.bf)):
            return False
        else:
            tp = m / log(m)
        if m != d > tp:
            return True
        else:
            return False

    def __get_bf(self, m):  # return the branching factor of the mother node
        keys = m.keys()
        n_keys = len(keys)
        for k in keys:
            if k in self.special_tokens:
                n_keys -= 1
        return n_keys

    def __build_vocab_lookup(self):
        self.vocab_to_id = {}

        def traverse(trie, path):
            for k, v in trie.items():
                if k == 'IDX':
                    token = ''.join(path)
                    self.vocab_to_id[token] = v
                elif isinstance(v, dict):
                    traverse(v, path + [k])

        traverse(self.roots, [])
        self.vocab_to_id = dict(sorted(self.vocab_to_id.items(), key=lambda item: item[1]))
        self.vocab_to_id = {k.replace("[RSX]", ""): v for k, v in self.vocab_to_id.items()}
        self.id_to_vocab = {v: k for k, v in self.vocab_to_id.items()}

    def __build_vocab_freq(self):
        vocab_to_freq = {}  # Only store items with both IDX and ##

        def traverse(trie, path):
            # Check if current node has both IDX and ##
            if isinstance(trie, dict) and 'IDX' in trie and '##' in trie:
                token = ''.join(path)
                vocab_to_freq[token] = trie['##']  # Store frequency

            # Continue traversing
            for k, v in trie.items():
                if k not in ['IDX', '##'] and isinstance(v, dict):
                    traverse(v, path + [k])

        traverse(self.roots, [])

        # Sort by frequency (descending order - most frequent first)
        self.vocab_to_freq = dict(sorted(vocab_to_freq.items(), key=lambda item: item[1], reverse=True))

    def __sort_trie_by_freq(self, d):
        if not isinstance(d, dict):
            return d
        # Sort the dictionary items by the value of the nested key '##'
        sorted_items = sorted(
            d.items(),
            key=lambda item: item[1].get('##', float('-inf')) if isinstance(item[1], dict) else float('-inf'),
            reverse=True
        )
        # Clear the dictionary and update with sorted items
        d.clear()
        for k, v in sorted_items:
            d[k] = self.__sort_trie_by_freq(v)
        return d

    def __find_idx_path(self, d, target_value, path=None):
        if path is None:
            path = []
        for key, value in d.items():
            if key == 'IDX' and value == target_value:
                return path
            elif isinstance(value, dict):
                result = self.__find_idx_path(value, target_value, path + [key])
                if result is not None:
                    return result
        return None

    def __count_trie_nodes(self, trie_node):
        """
        Count the total number of nodes in a trie structure.

        Args:
            trie_node (dict): A dictionary representing a trie node

        Returns:
            int: Total number of nodes in the trie
        """
        if not isinstance(trie_node, dict):
            return 0

        # Count current node
        count = 1

        # Recursively count child nodes (excluding "IDX" which is not a child node)
        for key, value in trie_node.items():
            if key != "IDX" and key != "##" and isinstance(value, dict):
                count += self.__count_trie_nodes(value)

        return count

    def prepare_encoding(self):
        self.ids.append(0)
        self.tokens.append("")

    def train(self, training_corpus_path: str, text_column: str = "text"):
        """
        Train MorPiece tokenizer on your corpus

        Parameters
        ----------
        training_corpus_path : str / parquet file (content must be in the "text" column)
            path to training corpus directory
        """

        all_contents = ""
        n_files = 0

        training_corpus_path = str(training_corpus_path)
        if training_corpus_path.endswith(".parquet"):
            # --- Parquet processing ---
            try:
                import pandas as pd
            except ImportError:
                raise ImportError("pandas is required to read parquet files: pip install pandas pyarrow")
            df = pd.read_parquet(training_corpus_path)
            if text_column not in df.columns:
                raise ValueError(
                    f"Column '{text_column}' not found in parquet file. "
                    f"Available columns: {df.columns.tolist()}"
                )
            for content in df[text_column].dropna().astype(str):
                if self.use_tokenizers_lib:
                    content = self._preprocess_text(content)
                all_contents += content + "\n"
                n_files += 1
            print(f"MorPiece tokenizer training: loaded {n_files} rows from parquet '{training_corpus_path}' (column='{text_column}')...")
        else:
            # --- Simple text processing ---
            for filename in os.listdir(training_corpus_path):
                file_path = os.path.join(training_corpus_path, filename)
                if os.path.isfile(file_path):
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        if self.use_tokenizers_lib:
                            content = self._preprocess_text(content)
                        all_contents += content + "\n"
                        n_files += 1

        words = all_contents.split()
        print(f"MorPiece tokenizer training: processing text in corpus ({n_files} files, {len(words)} tokens)...")
        if (self.ooa):
            print('Saving Order Of Acquisition Vocabulary every 100000 tokens (.):')

         # --- type-frequency counting pass (always needed for self.types) ---
        for word in words:
            word_alpha = ''.join([char for char in word if char.isalpha() or char == "'" or char == "-"])
            word = word_alpha if word_alpha else ''.join([char for char in word])
            if word:
                if word not in self.types:
                    self.types[word] = 1
                else:
                    self.types[word] += 1
                self.num_tokens_in_corpus += 1
                self.num_chars_in_corpus += len(word)

        if self.type_based:
            # In type-based mode each trie path is built/traversed exactly once,
            # regardless of how many times the word occurs in the corpus.
            # The corpus token counts above are still recorded normally so that
            # statistics (TTR, compression ratio, …) remain meaningful.
            print(f"  type_based=True: building tries from {len(self.types)} unique types "
                  f"(corpus had {self.num_tokens_in_corpus} tokens, {len(self.types)} types, "
                  f"TTR={round(len(self.types)/self.num_tokens_in_corpus, 3)})")
            training_items = self.types.keys()   # iterate over types only
        else:
            training_items = words               # iterate over every token

        token_counter = 0
        for word in training_items:
            word_alpha = ''.join([char for char in word if char.isalpha() or char == "'" or char == "-"])
            word = word_alpha if word_alpha else ''.join([char for char in word])
            if word:
                self.current_world = word  # for debug_trie
                self.__build_trie(word[::-1], self.infls)  # create inflections trie
                self.__build_trie(word, self.roots)        # create roots trie
                self.__morsplit(word)
                token_counter += 1

                if self.ooa and not self.type_based:
                    # ooa snapshots are token-exposure-based; skip in type_based mode
                    if token_counter % 100000 == 0:
                        self.__incremental_cleaning(self.roots, self.min_frequency)
                        self.__build_vocab_freq()
                        ooa_dir = os.path.join('ooa_bf' + str(self.bf) + '_cutoff' + str(self.cutoff) + '_mfreq' + str(self.min_frequency))
                        print('.', end='')
                        os.makedirs("tokenizer/" + ooa_dir, exist_ok=True)
                        self.save_vocab("tokenizer/" + ooa_dir + "/vocab_" + str(token_counter) + ".json")
        if self.ooa and self.type_based:
            print("  (ooa snapshots skipped in type_based mode — no meaningful token-exposure axis)")

        self.types = dict(sorted(self.types.items(), key=lambda item: item[1], reverse=True))
        self.__sort_trie_by_freq(self.roots)
        self.num_chars_in_trie = self.__count_trie_nodes(self.roots)
        if self.ooa:
            print("")
        print(
            f"MorPiece tokenizer training: trie optimization (vocab_size={self.vocab_size}, min_freq={self.min_frequency}, cutoff={self.cutoff}, bf={self.bf})...")
        self.__optimize(self.roots)
        self.num_chars_in_optimized_trie = self.__count_trie_nodes(self.roots)

        print(f"MorPiece tokenizer trained: final vocabulary created with {self.get_vocab_size()} tokens")
        if self.ooa:
            print(f"MorPiece tokenizer Order of Acquisition of each morphological split prepared.")

    def encode(self, sentence: str):
        """Encode a sentence into token IDs and tokens"""
        # Apply preprocessing if tokenizers lib is enabled
        if self.use_tokenizers_lib:
            sentence = self._preprocess_text(sentence)

        self.ids, self.tokens = [], []
        words = sentence.strip().split()

        for word in words:
            if word in self.roots['[RSX]']:
                self.ids.append(self.roots['[RSX]'][word]['IDX'])
                self.tokens.append(word)
            else:
                self.prepare_encoding()
                self.__retrieve(word, self.roots)

        # Apply post-processing if tokenizers lib is enabled
        if self.use_tokenizers_lib:
            self.ids = self._postprocess_tokens(self.ids)

        return self.ids, self.tokens

    def decode(self, sentence_idxs):
        """Decode token IDs back to tokens"""
        tokens = []
        for idx in sentence_idxs:
            token = self.id_to_vocab.get(idx, '<unk>')
            tokens.append(token)

        # Apply decoding if tokenizers lib is enabled
        if self.use_tokenizers_lib and tokens:
            # Join tokens and apply decoder
            text = ''.join(tokens)
            try:
                decoded_text = self.decoder.decode(text)
                return decoded_text.split()
            except:
                # Fallback to original tokens if decoding fails
                return tokens

        return tokens

    def set_special_tokens(self, token_list):
        for item in token_list:
            if item not in self.roots['[RSX]'].keys():
                self.roots['[RSX]'][item] = {'IDX': None}
                self.roots['[RSX]'][item]['IDX'] = self.idx
                self.idx += 1

    def pad_sentence(self, sentence: str, l: int, pad: str = '<pad>'):
        """
        Pads the given sentence with "[pad]" tokens at the beginning to reach the desired length.

        Parameters:
        - sentence (str): The original sentence to be padded.
        - pad (str): pad special token (default <pad>).
        - l (int): The desired total number of tokens in the sentence after padding.

        Returns:
        - str: The padded sentence.
        """
        words = sentence.split()
        n_pad = max(l - len(words), 0)  # Ensure n_pad is not negative
        pad_tokens = [pad] * n_pad
        padded_sentence = ' '.join(pad_tokens + words)
        return padded_sentence

    def get_num_chars_in_trie(self):
        return self.num_chars_in_trie

    def get_num_chars_in_optimized_trie(self):
        return self.num_chars_in_optimized_trie

    def get_num_chars_in_corpus(self):
        return self.num_chars_in_corpus

    def get_vocab_size(self) -> int:
        self.vocab_size = self.idx
        return self.idx

    def get_num_tokens_in_corpus(self):
        return self.num_tokens_in_corpus

    def get_num_types_in_corpus(self):
        return len(self.types)

    def get_compression_ratio(self):
        return round(self.num_chars_in_optimized_trie / self.num_chars_in_trie, 3)

    def get_ttr(self):
        return round(len(self.types) / self.num_tokens_in_corpus, 3)

    def from_pretrained(self, load_file):
        """
        Load a native MorPiece tokenizer from *load_file*/tokenizer.json.

        Fixes vs original
        -----------------
        • Uses os.path.join (no hardcoded '/' separator).
        • Restores self.idx from the maximum vocab ID so that
          get_vocab_size() returns the correct value after loading.
        • Restores special-token IDs from the 'special_token_ids' key
          written by save_pretrained() v1.3+ — falls back silently to the
          __init__ defaults for older checkpoints.
        """
        tokenizer_path = os.path.join(load_file, 'tokenizer.json')
        print(f"Loading MorPiece tokenizer from {tokenizer_path}...")
        with open(tokenizer_path, 'r', encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and 'roots' in data:
            self.roots       = data['roots']
            self.vocab_to_id = data.get('vocab', {})
            self.id_to_vocab = {v: k for k, v in self.vocab_to_id.items()}

            # Restore idx so get_vocab_size() is correct
            if self.vocab_to_id:
                self.idx = max(self.vocab_to_id.values()) + 1

            # Restore special-token IDs if persisted (v1.3+)
            if 'special_token_ids' in data:
                sp = data['special_token_ids']
                self.unk_token_id  = sp.get('unk',  self.unk_token_id)
                self.pad_token_id  = sp.get('pad',  self.pad_token_id)
                self.bos_token_id  = sp.get('bos',  self.bos_token_id)
                self.eos_token_id  = sp.get('eos',  self.eos_token_id)
                self.mask_token_id = sp.get('mask', self.mask_token_id)
        else:
            # Old format (tokenizer.json only had roots, no vocab key)
            self.roots       = data
            self.vocab_to_id = {}
            self.id_to_vocab = {}

        if '[RSX]' not in self.roots:
            raise ValueError("Invalid tokenizer format: Missing [RSX] root node.")

    def save_pretrained(self, save_file):
        """
        Save the native MorPiece format (roots + vocab + special_token_ids).
        Use save_HF() to produce the HuggingFace-compatible format for inference.

        Parameters
        ----------
        save_file : str
            Full path to the output JSON file  (e.g. 'output/native/tokenizer.json')
        """
        self.__build_vocab_lookup()
        with open(save_file, 'w', encoding='utf-8') as f:
            json.dump({
                'roots': self.roots,
                'vocab': self.vocab_to_id,
                # Persist IDs so from_pretrained() never relies on __init__ defaults
                'special_token_ids': {
                    'unk':  self.unk_token_id,
                    'pad':  self.pad_token_id,
                    'bos':  self.bos_token_id,
                    'eos':  self.eos_token_id,
                    'mask': self.mask_token_id,
                },
            }, f, indent=2)

    def save_vocab(self, save_file):
        self.__sort_trie_by_freq(self.roots)
        self.__build_vocab_lookup()
        with open(save_file, 'w') as f:
            json.dump({
                'vocab': self.vocab_to_freq
            }, f, indent=2)

    def save_types(self, file):
        with open(file, 'w') as f:
            json.dump(self.types, f, indent=2)

    def save_ooa(self, file):
        """
        After training, this saves the Order of Acquisition of each morphological split assumed with indication of the token exposure received and the actual lexicon in a tab separated value file.

        current word, morphological piece detached, vocabulary size, token exposure].
        """
        with open(file, 'w', encoding='utf8') as f:
            for row in self.ooa_data:
                f.write('\t'.join(str(cell) for cell in row) + '\n')

    def save_HF(self, save_directory, model_max_length=1024):
        """
        Save the MorPiece tokenizer in HuggingFace format.

        This is the **primary format for inference and for the GPT-2 training
        pipeline** — load with PreTrainedTokenizerFast.from_pretrained(save_directory).

        The tokenizer.json uses a WordPiece model with '++' as the continuing-
        subword prefix, matching MorPiece's suffix notation.  Tokenization is
        functionally equivalent to the native encode() for the vast majority of
        inputs and is backed by Rust (orders of magnitude faster than the Python
        trie traversal).

        Fixes vs original
        -----------------
        • model_max_length is a parameter (was hardcoded to 512).
        • Normalizer now includes Lowercase (matching the training normalizer),
          so the HF tokenizer lowercases inputs consistently with how the
          vocabulary was built.
        • tokenizer_config.json cleaned up: BERT-specific fields removed
          (do_lower_case, do_basic_tokenize, never_split, tokenize_chinese_chars,
          auto_map, sep_token, cls_token, special_tokens_map_file, name_or_path);
          added_tokens_decoder added for correct special-token recognition.

        Parameters
        ----------
        save_directory  : str   Directory to write tokenizer files into.
        model_max_length: int   Maximum sequence length (default 1024).
        """
        os.makedirs(save_directory, exist_ok=True)

        if not hasattr(self, 'vocab_to_id') or self.vocab_to_id is None:
            self.__build_vocab_lookup()

        # Build vocab list ordered by token ID
        vocab = [""] * len(self.vocab_to_id)
        for token, token_id in self.vocab_to_id.items():
            if token_id < len(vocab):
                vocab[token_id] = token

        # Fill any gaps with <unk>
        for i in range(len(vocab)):
            if vocab[i] == "":
                vocab[i] = "<unk>"

        # Special-token ID map (read from instance so it survives from_pretrained)
        sp = {
            "<unk>":  self.unk_token_id,   # 0
            "<pad>":  self.pad_token_id,   # 1
            "<s>":    self.bos_token_id,   # 2
            "</s>":   self.eos_token_id,   # 3
            "<mask>": self.mask_token_id,  # 4
        }

        # Pin special tokens at their correct positions in the vocab list
        for token, token_id in sp.items():
            if token_id < len(vocab):
                vocab[token_id] = token

        # ---- tokenizer.json -----------------------------------------------
        tokenizer_json = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [
                {"id": sp[tok], "content": tok, "single_word": False,
                 "lstrip": False, "rstrip": False, "normalized": False, "special": True}
                for tok in ("<unk>", "<pad>", "<s>", "</s>", "<mask>")
            ],
            # Normalizer matches MorPiece training: lowercase first, then NFKC.
            # Without Lowercase here the HF tokenizer would produce <unk> for
            # any uppercase input, because the vocabulary only contains lowercase
            # tokens (the training normalizer lowercased everything).
            "normalizer": {
                "type": "Sequence",
                "normalizers": [
                    {"type": "Lowercase"},
                    {"type": "NFKC"},
                ]
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
            "decoder": {
                "type":    "WordPiece",
                "prefix":  "++",
                "cleanup": True,
            },
            "model": {
                "type":                      "WordPiece",
                "unk_token":                 "<unk>",
                "continuing_subword_prefix": "++",
                "max_input_chars_per_word":  100,
                "vocab": {token: i for i, token in enumerate(vocab) if token.strip() != ""},
            },
        }

        tokenizer_json_path = os.path.join(save_directory, "tokenizer.json")
        with open(tokenizer_json_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_json, f, indent=2, ensure_ascii=False)

        # ---- tokenizer_config.json ----------------------------------------
        # Clean HF format — no BERT-specific fields (do_lower_case,
        # do_basic_tokenize, never_split, etc. are irrelevant for
        # PreTrainedTokenizerFast and can confuse the loader).
        added_tokens_decoder = {}
        for token, token_id in sp.items():
            added_tokens_decoder[str(token_id)] = {
                "content":     token,
                "lstrip":      False,
                "normalized":  False,
                "rstrip":      False,
                "single_word": False,
                "special":     True,
            }

        tokenizer_config = {
            "added_tokens_decoder":         added_tokens_decoder,
            "bos_token":                    "<s>",
            "clean_up_tokenization_spaces": False,
            "eos_token":                    "</s>",
            "extra_special_tokens":         {},
            "mask_token":                   "<mask>",
            "model_max_length":             model_max_length,
            "pad_token":                    "<pad>",
            "tokenizer_class":              "PreTrainedTokenizerFast",
            "unk_token":                    "<unk>",
        }

        config_path = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)

        # ---- special_tokens_map.json --------------------------------------
        special_tokens_map = {
            "bos_token":  "<s>",
            "eos_token":  "</s>",
            "unk_token":  "<unk>",
            "pad_token":  "<pad>",
            "mask_token": "<mask>",
        }
        with open(os.path.join(save_directory, "special_tokens_map.json"),
                  'w', encoding='utf-8') as f:
            json.dump(special_tokens_map, f, indent=2, ensure_ascii=False)

        # ---- vocab.txt (for human inspection) ----------------------------
        with open(os.path.join(save_directory, "vocab.txt"), 'w', encoding='utf-8') as f:
            for token in vocab:
                if token.strip() != "":
                    f.write(f"{token}\n")

        print(f"MorPiece tokenizer saved in HuggingFace format → {save_directory}")
        print(f"  tokenizer.json  tokenizer_config.json  "
              f"special_tokens_map.json  vocab.txt")
        print(f"  vocab_size={len([t for t in vocab if t.strip()])}  "
              f"model_max_length={model_max_length}")
        print(f"\nLoad: PreTrainedTokenizerFast.from_pretrained('{save_directory}')")

    def create_WordPiece_tokenizer(self):
        """
        Create a complete tokenizers.Tokenizer instance with all components integrated.
        This can be used for more advanced tokenization tasks.
        """
        if not self.use_tokenizers_lib:
            print("Tokenizers library integration is disabled. Enable it in the constructor.")
            return None

        from tokenizers import Tokenizer
        from tokenizers.models import WordPiece

        # Build vocab lookup if not already done
        if not hasattr(self, 'vocab_to_id') or self.vocab_to_id is None:
            self.__build_vocab_lookup()

        # Create the vocabulary list ordered by token IDs
        vocab = [""] * len(self.vocab_to_id)
        for token, token_id in self.vocab_to_id.items():
            if token_id < len(vocab):
                vocab[token_id] = token

        # Fill any gaps with <unk> token
        for i in range(len(vocab)):
            if vocab[i] == "":
                vocab[i] = "<unk>"

        # Create vocab dict for WordPiece model
        vocab_dict = {token: i for i, token in enumerate(vocab)}

        # Create tokenizer with WordPiece model
        tokenizer = Tokenizer(WordPiece(vocab_dict, unk_token="<unk>"))

        # Apply all the components
        tokenizer.normalizer = self.normalizer
        tokenizer.pre_tokenizer = self.pre_tokenizer
        tokenizer.decoder = self.decoder
        tokenizer.post_processor = self.post_processor

        tokenizer.save()
        return tokenizer

    def create_bpe_tokenizer(self):
        """
        Create a complete tokenizers.Tokenizer instance with BPE model using the trained vocabulary.
        This creates a BPE tokenizer based on the morphological splits learned during training.

        Returns:
            tokenizers.Tokenizer: A complete BPE tokenizer instance, or None if tokenizers lib is disabled
        """
        if not self.use_tokenizers_lib:
            print("Tokenizers library integration is disabled. Enable it in the constructor.")
            return None

        try:
            from tokenizers import Tokenizer
            from tokenizers.models import BPE
        except ImportError:
            print("tokenizers library not available. Install with: pip install tokenizers")
            return None

        # Build vocab lookup if not already done
        if not hasattr(self, 'vocab_to_id') or self.vocab_to_id is None:
            self.__build_vocab_lookup()

        if not self.vocab_to_id:
            print("No vocabulary found. Please train the tokenizer first.")
            return None

        # Create the vocabulary list ordered by token IDs
        vocab = [""] * len(self.vocab_to_id)
        for token, token_id in self.vocab_to_id.items():
            if token_id < len(vocab):
                vocab[token_id] = token

        # Fill any gaps with <unk> token
        for i in range(len(vocab)):
            if vocab[i] == "":
                vocab[i] = "<unk>"

        # For BPE, we need to create merges based on the morphological structure
        # Extract merges from the trie structure
        merges = []
        vocab_set = set(vocab)

        def extract_merges_from_trie(trie, path=""):
            """Extract potential merges from the trie structure"""
            if not isinstance(trie, dict):
                return

            for key, subtrie in trie.items():
                if key in ['IDX', '##'] or key in self.reserved_keys:
                    continue

                current_path = path + key

                # If this path exists in vocab and has children, create merges
                if current_path in vocab_set and isinstance(subtrie, dict):
                    for child_key, child_subtrie in subtrie.items():
                        if child_key not in ['IDX', '##'] and child_key not in self.reserved_keys:
                            child_path = current_path + child_key
                            if child_path in vocab_set:
                                # Create a merge: "current_path child_key" -> "child_path"
                                merge_pair = (current_path, child_key)
                                if merge_pair not in merges:
                                    merges.append(merge_pair)

                # Recursively process subtrie
                extract_merges_from_trie(subtrie, current_path)

        # Extract merges from roots (excluding special tokens section)
        for root_key, root_trie in self.roots.items():
            if root_key != '[RSX]':  # Skip special tokens
                extract_merges_from_trie(root_trie)

        # Create vocab dict for BPE model (BPE expects string keys)
        vocab_dict = {token: i for i, token in enumerate(vocab)}

        # Create BPE model
        # Note: BPE requires merges to be in a specific format
        bpe_merges = []
        for merge in merges[:1000]:  # Limit merges to prevent issues
            if len(merge) == 2:
                bpe_merges.append(" ".join(merge))

        try:
            # Create tokenizer with BPE model
            bpe_model = BPE(vocab=vocab_dict, merges=bpe_merges, unk_token="<unk>")
            tokenizer = Tokenizer(bpe_model)

            # Apply all the preprocessing components
            tokenizer.normalizer = self.normalizer
            tokenizer.pre_tokenizer = self.pre_tokenizer
            tokenizer.decoder = self.decoder
            tokenizer.post_processor = self.post_processor

            print(f"BPE tokenizer created with {len(vocab_dict)} vocabulary items and {len(bpe_merges)} merges")
            return tokenizer

        except Exception as e:
            print(f"Error creating BPE tokenizer: {e}")
            # Fallback: create simple BPE with minimal merges
            try:
                simple_bpe = BPE(vocab=vocab_dict, merges=[], unk_token="<unk>")
                tokenizer = Tokenizer(simple_bpe)
                tokenizer.normalizer = self.normalizer
                tokenizer.pre_tokenizer = self.pre_tokenizer
                tokenizer.decoder = self.decoder
                tokenizer.post_processor = self.post_processor
                print(f"Created simplified BPE tokenizer with {len(vocab_dict)} vocabulary items")
                return tokenizer
            except Exception as fallback_e:
                print(f"Failed to create BPE tokenizer: {fallback_e}")
                return None
