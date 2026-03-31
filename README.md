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
- Type-based approach, intead of token-based approach
