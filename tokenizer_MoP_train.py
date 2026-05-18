import os
import argparse
import tokenizer_MorPiece as MoP
from collections import Counter
from pathlib import Path

def calculate_stats(tokenizer, args):
    counter, n_words, n_tokens = Counter(), 0, 0
    n_unk, n_valid_pieces = 0, 0
    example_printed = False
    training_path = str(args.training_dir)

    if training_path.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(training_path)
        if args.text_column not in df.columns:
            raise ValueError(
                f"Column '{args.text_column}' not found in parquet file. "
                f"Available columns: {df.columns.tolist()}"
            )
        texts = df[args.text_column].dropna().astype(str).tolist()
        for text in texts:
            text = text.strip()
            if len(text) > 0:
                n_words += len(text.split())
                ids, tokens = tokenizer.encode(text)
                n_tokens += len(tokens)
                counter.update(tokens)
                n_unk += tokens.count('<unk>')
                n_valid_pieces += sum(1 for t in tokens if t.startswith('++'))
                if not example_printed:
                    print("Example of tokenization:")
                    print(text[:200])
                    print(tokenizer.decode(ids)[:200])
                    example_printed = True
    else:
        for file_path in sorted(args.training_dir.glob("*")):
            if file_path.is_file():
                for i, document in enumerate(file_path.open("r")):
                    text = document.strip()
                    if len(text) > 0:
                        n_words += len(text.split())
                        ids, tokens = tokenizer.encode(text)
                        n_tokens += len(tokens)
                        counter.update(tokens)
                        n_unk += tokens.count('<unk>')
                        n_valid_pieces += sum(1 for t in tokens if t.startswith('++'))
                        if i == 100:
                            print("Example of tokenization:")
                            print(text)
                            print(tokenizer.decode(ids))

    freqs = sorted(counter.values())
    if not freqs:
        print("Warning: no tokens collected in calculate_stats — skipping statistics.")
        return

    f_95 = freqs[min(int(len(freqs) * 0.95), len(freqs) - 1)]
    f_50 = freqs[min(len(freqs) // 2, len(freqs) - 1)]
    singleton_ratio = sum(1 for f in counter.values() if f == 1) / len(counter)
    avg_tokens_per_word = n_tokens / n_words if n_words > 0 else 0

    # OOV rate: fraction of all output tokens that are <unk>
    oov_rate = n_unk / n_tokens if n_tokens > 0 else 0
    # Morphological piece rate: fraction of output tokens that are valid ++ suffixes
    piece_rate = n_valid_pieces / n_tokens if n_tokens > 0 else 0
    # Signal-to-noise: valid morphological pieces per <unk> (higher is better)
    piece_to_unk = (n_valid_pieces / n_unk) if n_unk > 0 else float('inf')

    top_100_tokens = counter.most_common(100)

    print(f"F_{{95%}} is {f_95}")
    print(f"F_{{50%}} is {f_50}")
    print(f"Hapax ratio is {singleton_ratio:.4f}")
    print(f"Average tokens per word is {avg_tokens_per_word:.4f}")
    print(f"OOV rate (<unk>/total tokens): {oov_rate:.4f}  ← lower is better")
    print(f"Morphological piece rate (++/total tokens): {piece_rate:.4f}  ← higher means more segmentation")
    print(f"Piece-to-unk ratio (++/<unk>): {piece_to_unk:.2f}  ← higher is better")

    stats_path = os.path.join(args.output_dir, "stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"Vocabulary size: {args.vocab_size}\n")
        f.write(f"F_{{95%}} is {f_95}\n")
        f.write(f"F_{{50%}} is {f_50}\n")
        f.write(f"Hapax ratio is {singleton_ratio:.4f}\n")
        f.write(f"Average tokens per word is {avg_tokens_per_word:.4f}\n")
        f.write(f"OOV rate (<unk>/total tokens): {oov_rate:.4f}\n")
        f.write(f"Morphological piece rate (++/total tokens): {piece_rate:.4f}\n")
        f.write(f"Piece-to-unk ratio (++/<unk>): {piece_to_unk:.2f}\n")
        f.write(f"First most frequent 100 tokens:\n{top_100_tokens}\n")
        
def main():
    # Data arguments and Tokenizer's parameters
    parser = argparse.ArgumentParser(description='Train MorPiece Tokenizer with a custom corpus')
    parser.add_argument('--training_dir', type=Path, required=False, default='./corpus',
                        help='Path to the corpus files to be used for training')
    parser.add_argument('--output_dir', type=str, required=False, default='./tokenizer',
                        help='Path to the folder for saving the trained tokenizer')
    parser.add_argument('--min_frequency', type=int, required=False, default=10,
                        help='Min frequency to include a token in the final vocabulary')
    parser.add_argument('--cutoff', type=int, required=False, default=100,
                        help='Cutoff parameter expressing the lower frequency threshold (for mother node) to apply the Tolerance Principle')
    parser.add_argument('--bf', type=int, required=False, default=10,
                        help='Branching Factor expressing the max number of daughters needed to apply the Tolerance Principle')
    parser.add_argument('--vocab_size', type=int, required=False, default=60000, help='Max final vocabulary size')
    parser.add_argument('--ooa', action='store_true',
                        help='If True, the Order of Acquisition of various splits is generated (this requires extra computation)')
    parser.add_argument('--type_based', action='store_true',
                        help='If True, the Order of Acquisition of various splits is generated (this requires extra computation)')
    parser.add_argument('--text_column', type=str, required=False, default='text',
                        help='Column name for text content when training_dir is a .parquet file (default: text)')

    args = parser.parse_args()

    corpus_file = args.training_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Tokenizer training
    print(
        f"Parameters: vocab_size={args.vocab_size}, cutoff={args.cutoff}, min_frequency={args.min_frequency}, bf={args.bf}, ooa={args.ooa}")
    mp = MoP.MorPiece(vocab_size=args.vocab_size, cutoff=args.cutoff, min_frequency=args.min_frequency, bf=args.bf,
                      ooa=args.ooa, type_based=args.type_based, use_tokenizers_lib=True)

    mp.train(corpus_file, text_column=args.text_column)

    # -----------------------------------------------------------------------
    # PRIMARY SAVE: HuggingFace format directly in output_dir/
    # To load this via
    #   PreTrainedTokenizerFast.from_pretrained(output_dir)
    # -----------------------------------------------------------------------
    mp.save_HF(output_dir, model_max_length=1024)

    # -----------------------------------------------------------------------
    # NATIVE SAVE: MorPiece roots + vocab + special_token_ids
    # Needed only if you want to reload the trie for further training,
    # inspection, or running tokenizer_MoP_train.py stats on a saved model.
    # -----------------------------------------------------------------------
    native_dir = os.path.join(output_dir, 'native')
    os.makedirs(native_dir, exist_ok=True)
    mp.save_pretrained(os.path.join(native_dir, 'tokenizer.json'))
    mp.save_types(os.path.join(native_dir, 'vocab_types.json'))
    if (args.ooa):
        ooa_dir = os.path.join(output_dir, 'ooa_bf' + str(args.bf) + '_cutoff' + str(args.cutoff) + '_mfreq' + str(
            args.min_frequency))
        os.makedirs(ooa_dir, exist_ok=True)
        mp.save_ooa(os.path.join(ooa_dir, 'ooa_vocab.tsv'))
    # To reload a previously saved tokenizer (native format):
    # mp.from_pretrained(os.path.join(output_dir, 'native'))
    # To reload as PreTrainedTokenizerFast (HF format, for inference):
    # from transformers import PreTrainedTokenizerFast
    # tokenizer = PreTrainedTokenizerFast.from_pretrained(output_dir)

    # Tokenizer training information
    print(f"Number of tokens processed: {mp.get_num_tokens_in_corpus()}")
    print(f"Number of types processed: {mp.get_num_types_in_corpus()}")
    print(f"Number of chars in the root trie before optimization: {mp.get_num_chars_in_trie()}")
    print(f"Number of chars in the root trie after optimization: {mp.get_num_chars_in_optimized_trie()}")
    print(f"Compression ratio: {mp.get_compression_ratio()}")
    print(f"Vocabulary size: {mp.vocab_size}")
    print(f"Most common tokens: {list(mp.types.items())[:20]}")
    print(f"Model and tokenizer in {output_dir}")
    
    print("\nCalculating BPE-like statistics...")
    calculate_stats(mp, args)

    # to test the tokenizer
    # s = "the worker must go in the office where the dogs barks all days - long much longer than any impossible cats"
    # s = "carlo correva a prenderlo , io burpo , egli burpa , noi sbirpiamo , sdupolarlo , io mangio tu mangi egli mangia noi mangiamo voi mangiate loro mangiano mario corse loro corsero improbabile che loro siano andate al maremoto io lavo tu lavi lui lava noi laviamo voi lavate loro lavano lavarono laveranno laverebbero lavarlo"
    s =  """Who have those men revealed they helped?
            Leslie remembered some guest that has bothered women.
            Mark figured out that most governments appreciate Steve.
            Dennis has seen this tooth that Kristin wasn't concealing that is astounding men.
            Every association figured out that most drivers that forfeit investigated Irene.
            A lady has remembered who the actors conceal.""" 
    print("Sentence to tokenize: " + s)
    ids, tokens = mp.encode(s)

    # token_texts, token_ids = [token_text, token_id for token_text, token_id in mp.tokenized_words]

    print(ids, tokens)
    print(mp.decode(ids))


if __name__ == "__main__":
    main()
