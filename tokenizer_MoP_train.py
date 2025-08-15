import os
import argparse
import tokenizer_MorPiece as MoP


def main():
    # Data arguments and Tokenizer's parameters
    parser = argparse.ArgumentParser(description='Train MorPiece Tokenizer with a custom corpus')
    parser.add_argument('--training_dir', type=str, required=False, default='./corpus',
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

    args = parser.parse_args()

    corpus_file = args.training_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Tokenizer training
    print(
        f"Parameters: vocab_size={args.vocab_size}, cutoff={args.cutoff}, min_frequency={args.min_frequency}, bf={args.bf}, ooa={args.ooa}")
    mp = MoP.MorPiece(vocab_size=args.vocab_size, cutoff=args.cutoff, min_frequency=args.min_frequency, bf=args.bf,
                      ooa=args.ooa)

    mp.train(corpus_file)
    mp.save_pretrained(os.path.join(output_dir, 'tokenizer.json'))
    mp.save_types(os.path.join(output_dir, 'vocab_types.json'))
    if (args.ooa):
        ooa_dir = os.path.join(output_dir, 'ooa_bf' + str(args.bf) + '_cutoff' + str(args.cutoff) + '_mfreq' + str(
            args.min_frequency))
        os.makedirs(ooa_dir, exist_ok=True)
        mp.save_ooa(os.path.join(ooa_dir, 'ooa_vocab.tsv'))
    # Uncomment the following line and comment the previous three lines to load a pre-trained tokenizer
    # mp.from_pretrained(output_dir)
    mp.save_HF(output_dir + "/hf")
    ## mp = PreTrainedTokenizerFast.from_pretrained(output_dir + "/hf")

    # Tokenizer training information
    print(f"Number of tokens processed: {mp.get_num_tokens_in_corpus()}")
    print(f"Number of types processed: {mp.get_num_types_in_corpus()}")
    print(f"Number of chars in the root trie before optimization: {mp.get_num_chars_in_trie()}")
    print(f"Number of chars in the root trie after optimization: {mp.get_num_chars_in_optimized_trie()}")
    print(f"Compression ratio: {mp.get_compression_ratio()}")
    print(f"Vocabulary size: {mp.vocab_size}")
    print(f"Most common tokens: {list(mp.types.items())[:20]}")
    print(f"Model and tokenizer in {output_dir}")

    # to test the tokenizer
    # s = "the worker must go in the office where the dogs barks all days - long much longer than any impossible cats"
    # s = "carlo correva a prenderlo , io burpo , egli burpa , noi sbirpiamo , sdupolarlo , io mangio tu mangi egli mangia noi mangiamo voi mangiate loro mangiano mario corse loro corsero improbabile che loro siano andate al maremoto io lavo tu lavi lui lava noi laviamo voi lavate loro lavano lavarono laveranno laverebbero lavarlo"
    s = """interest in the chemistry of unhexquadium is largely prompted by predictions that the isotope 482uhq (with 164 protons and 318 neutrons ) , would be at the center of a possible second island of stability ( the
    # first being centered on 306ubb or 298fl ) ."""
    # s = "laveranno"
    print("Sentence to tokenize: " + s)
    ids, tokens = mp.encode(s)

    # token_texts, token_ids = [token_text, token_id for token_text, token_id in mp.tokenized_words]

    print(ids, tokens)
    print(mp.decode(ids))


if __name__ == "__main__":
    main()