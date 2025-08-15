__version__ = "1.2.1"
__author__ = "Cristiano Chesi & NeTS Lab @ IUSS"
__email__ = "cristiano.chesi@iusspavia.it"
__status__ = "Research"
__date__ = "2025-08-13"
__license__ = "MIT"

"""
MorPiece is a split-based tokenization library that incrementally chunks words into potentially meaningful morphemes. The splitting procedure consists of evaluating if the Tolerance Principle (Yang 2016) applies at every character every time an incoming word "traverses" the lexicon. 

Take the word "cats": a root "trie" (c->a->t->s) and a inflectional trie (s->t->a->c) are considered. 
"Traversing" the lexicon means adding 1 to each node counter that is traversed both in the root trie and in the inflectional trie. If a path does not exists, it is initialized to 1. 
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
    Token frequency is indicated at each node in the trie.
    The split procedure
    optimization procedure at the end of the training exclucded

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

    Methods
    -------
    train(corpus=None)
        train the tokenizer using a text file (UTF-8 encoding preferred)
    """

    ids: list[int]
    tokens: list[str]
    vocab_size: int

    def __init__(self, vocab_size=30000, min_frequency=10, cutoff=100, bf=10, special_tokens=None, ooa=True, use_tokenizers_lib=False):
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

    def __build_trie(self, wordpiece, root) -> None:
        if wordpiece[0] in root:
            root[wordpiece[0]]['##'] += 1
        else:
            root[wordpiece[0]] = {}
            root[wordpiece[0]]['##'] = 1
        if len(wordpiece) > 1:
            self.__build_trie(wordpiece[1:], root[wordpiece[0]])
        else:
            if 'IDX' not in root[wordpiece[0]]:
                root[wordpiece[0]]['IDX'] = 1

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

    def __find_path(self, word, trie):
        self.node = trie[word[0]]
        self.m_freq = self.d_freq
        self.d_freq = trie[word[0]]['##']
        self.m_daughters = len(trie[word[0]])
        if len(word) > 1:
            self.__find_path(word[1:], trie[word[0]])

    def __retrieve(self, string, trie):
        if string[0] in trie:
            self.tokens[-1] += string[0]
            if 'IDX' in trie[string[0]]:
                self.ids[-1] = trie[string[0]]['IDX']
            if len(string) > 1:
                self.__retrieve(string[1:], trie[string[0]])
        else:
            if string[0] in self.roots['++']:
                self.tokens.append('++' + string[0])
                self.ids.append(0)
                if 'IDX' in self.roots['++'][string[0]]:
                    self.ids[-1] = self.roots['++'][string[0]]['IDX']
                if len(string) > 1:
                    self.__retrieve(string[1:], self.roots['++'][string[0]])
            else:
                # self.tokenized_words.append(['<unk>', self.roots['[RSX]']['<unk>']['IDX']])
                self.ids.append(self.roots['[RSX]']['<unk>']['IDX'])
                self.tokens.append('<unk>')

    def __check_tp(self, m, d,
                   nd):  # verify if Tolerance Principle applies between m(other) and d(aughter) nodes, nd indicates the number of daughters
        if not ((m > self.cutoff) and (nd > self.bf)):
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

    def train(self, training_corpus_path: str):
        """
        Train MorPiece tokenizer on your corpus

        Parameters
        ----------
        training_corpus_path : str
            path to training corpus directory
        """

        all_contents = ""
        n_files = 0

        # Loop through files in the folder
        for filename in os.listdir(training_corpus_path):
            file_path = os.path.join(training_corpus_path, filename)

            # Only read if it's a file
            if os.path.isfile(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Apply preprocessing if tokenizers lib is enabled
                    if self.use_tokenizers_lib:
                        content = self._preprocess_text(content)
                    all_contents += content + "\n"
                    n_files += 1

        words = all_contents.split()
        print(f"MorPiece tokenizer training: processing text in corpus ({n_files} files, {len(words)} tokens)...")
        if (self.ooa):
            print('Saving Order Of Acquisition Vocabulary every 100000 tokens (.):')

        for word in words:
            word_alpha = ''.join([char for char in word if char.isalpha() or char == "'" or char == "-"])
            if not word_alpha:
                word = ''.join([char for char in word])
            else:
                word = word_alpha
            if word:
                self.current_world = word  # for debug_trie
                self.__build_trie(word[::-1], self.infls)  # create inflections trie
                self.__build_trie(word, self.roots)  # create roots trie
                self.__morsplit(word)
                if word not in self.types:  # count tokens and chars in corpus
                    self.types[word] = 1
                else:
                    self.types[word] += 1
                self.num_tokens_in_corpus += 1
                self.num_chars_in_corpus += len(word)

                if self.ooa:
                    if self.num_tokens_in_corpus % 100000 == 0:
                        self.__incremental_cleaning(self.roots, self.min_frequency)
                        self.__build_vocab_freq()
                        ooa_dir = os.path.join('ooa_bf' + str(self.bf) + '_cutoff' + str(self.cutoff) + '_mfreq' + str(self.min_frequency))
                        print('.', end='')
                        os.makedirs("tokenizer/" + ooa_dir, exist_ok=True)
                        self.save_vocab("tokenizer/" + ooa_dir + "/vocab_" + str(self.num_tokens_in_corpus) + ".json")

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
        print(f"Loading MorPiece tokenizer from {load_file}/tokenizer.json...")
        with open(load_file + '/tokenizer.json', 'r', encoding="utf-8") as f:
            data = json.load(f)

        # Backward compatibility: if old format, corpus is just roots
        if isinstance(data, dict) and 'roots' in data:
            self.roots = data['roots']
            self.vocab_to_id = data.get('vocab', {})  # fallback to empty dict if missing
            self.id_to_vocab = {v: k for k, v in self.vocab_to_id.items()}
        else:
            # Old format support (e.g., tokenizer.json only had roots)
            self.roots = data
            self.vocab_to_id = {}
            self.id_to_vocab = {}

        # Ensure [RSX] exists
        if '[RSX]' not in self.roots:
            raise ValueError("Invalid tokenizer format: Missing [RSX] root node.")

    def save_pretrained(self, save_file):
        self.__build_vocab_lookup()
        with open(save_file, 'w') as f:
            json.dump({
                'roots': self.roots,
                'vocab': self.vocab_to_id
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

    def save_HF(self, save_directory):
        """
        Save the MorPiece tokenizer in Hugging Face format to be loaded with PreTrainedTokenizerFast.

        Parameters:
        save_directory (str): Directory path where the tokenizer files will be saved
        """

        # Create directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)

        # Build vocab lookup if not already done
        if not hasattr(self, 'vocab_to_id') or self.vocab_to_id is None:
            self.__build_vocab_lookup()

        # Create the vocabulary list ordered by token IDs
        vocab = [""] * len(self.vocab_to_id)
        for token, token_id in self.vocab_to_id.items():
            if token_id < len(vocab):
                vocab[token_id] = token

        # Fill any gaps with <unk> token - this was problematic before
        for i in range(len(vocab)):
            if vocab[i] == "":
                vocab[i] = "<unk>"

        # Ensure special tokens are properly included in vocab if they exist
        special_tokens = {
            "<unk>": getattr(self, 'unk_token_id', 0),
            "<pad>": getattr(self, 'pad_token_id', 1),
            "<s>": getattr(self, 'bos_token_id', 2),
            "</s>": getattr(self, 'eos_token_id', 3),
            "<mask>": getattr(self, 'mask_token_id', 4)
        }

        # Make sure special tokens are in the vocab
        for token, token_id in special_tokens.items():
            if token_id < len(vocab):
                vocab[token_id] = token

        # Create the tokenizer.json in HuggingFace format with integrated components
        tokenizer_json = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [
                {
                    "id": special_tokens["<unk>"],
                    "content": "<unk>",
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True
                },
                {
                    "id": special_tokens["<pad>"],
                    "content": "<pad>",
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True
                },
                {
                    "id": special_tokens["<s>"],
                    "content": "<s>",
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True
                },
                {
                    "id": special_tokens["</s>"],
                    "content": "</s>",
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True
                },
                {
                    "id": special_tokens["<mask>"],
                    "content": "<mask>",
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True
                }
            ],
            # Fixed normalizer - removed problematic prepend
            "normalizer": {
                "type": "Sequence",
                "normalizers": [
                    {"type": "NFKC"}
                ]
            },
            # Re-enabled and fixed pre_tokenizer
            "pre_tokenizer": {
                "type": "Whitespace"
            },
            # Re-enabled post_processor with correct format
            "post_processor": {
                "type": "TemplateProcessing",
                "single": [
                    {"SpecialToken": {"id": "<s>", "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}}
                ],
                "pair": [
                    {"SpecialToken": {"id": "<s>", "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}},
                    {"SpecialToken": {"id": "<s>", "type_id": 1}},
                    {"Sequence": {"id": "B", "type_id": 1}}
                ],
                "special_tokens": {
                    "<s>": {
                        "id": "<s>",
                        "ids": [special_tokens["<s>"]],
                        "tokens": ["<s>"]
                    },
                    "</s>": {
                        "id": "</s>",
                        "ids": [special_tokens["</s>"]],
                        "tokens": ["</s>"]
                    }
                }
            },
            # Re-enabled decoder
            "decoder": {
                "type": "WordPiece",
                "prefix": "++",
                "cleanup": True
            },
            "model": {
                "type": "WordPiece",
                "unk_token": "<unk>",
                "continuing_subword_prefix": "++",
                "max_input_chars_per_word": 100,
                "vocab": {token: i for i, token in enumerate(vocab) if token.strip() != ""}  # Filter out empty tokens
            }
        }

        # Save tokenizer.json
        tokenizer_json_path = os.path.join(save_directory, "tokenizer.json")
        with open(tokenizer_json_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_json, f, indent=2, ensure_ascii=False)

        # Create tokenizer_config.json
        tokenizer_config = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "auto_map": {
                "AutoTokenizer": ["tokenizer.json", None]
            },
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "mask_token": "<mask>",
            "sep_token": "<sep>",
            "cls_token": "<cls>",
            "model_max_length": 512,
            "special_tokens_map_file": None,
            "name_or_path": save_directory,
            "tokenize_chinese_chars": True,
            "strip_accents": None,
            "do_lower_case": True,
            "do_basic_tokenize": True,
            "never_split": None
        }

        # Save tokenizer_config.json
        config_path = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)

        # Create special_tokens_map.json
        special_tokens_map = {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "sep_token": "<sep>",
            "pad_token": "<pad>",
            "cls_token": "<cls>",
            "mask_token": "<mask>"
        }

        # Save special_tokens_map.json
        special_tokens_path = os.path.join(save_directory, "special_tokens_map.json")
        with open(special_tokens_path, 'w', encoding='utf-8') as f:
            json.dump(special_tokens_map, f, indent=2, ensure_ascii=False)

        # Save vocab.txt (filter out empty tokens)
        vocab_txt_path = os.path.join(save_directory, "vocab.txt")
        with open(vocab_txt_path, 'w', encoding='utf-8') as f:
            for token in vocab:
                if token.strip() != "":  # Only write non-empty tokens
                    f.write(f"{token}\n")

        print(f"Tokenizer saved in HuggingFace format to: {save_directory}")
        print("Files created:")
        print(f"  - tokenizer.json")
        print(f"  - tokenizer_config.json")
        print(f"  - special_tokens_map.json")
        print(f"  - vocab.txt")
        print(f"\nTo load: tokenizer = PreTrainedTokenizerFast.from_pretrained('{save_directory}')")

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