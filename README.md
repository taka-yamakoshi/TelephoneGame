# TelephoneGame
informal description of the structure of this repo:

Since I cannot track all the data directories with git, I describe the intended directory structure here.

```
.
+-- model
|   +-- initial_sentences
|     +-- 12Tokens
|       +-- input.txt
|     +-- 21Tokens
|     +-- 37Tokens
+-- bert_corpus
|   +-- ...
|   +-- 11TokenSents: bert sentences generated with 11 tokens
|      +-- textfile
|         +-- bert-base-uncased
|            +-- 10_2000: batch_size and chain_len
|               +-- bert_gibbs_~.csv: output files from `bert.py`
|               +-- ...
|      +-- datafile
|         +-- bert-base-uncased: model name
|            +-- 5_51000_500_gibbs_mixture_random_mask_init_1_0
|               +-- POSFreqAllBert.pkl: output files from `CountFreq.py`
|               +-- ...
|            +-- ...
|   +-- 12TokenSents
|   +-- 21TokenSents
|   +-- 37TokenSents
|   +-- ...
+-- wikicorpus
    +-- prepared_wikipedia
    +-- TokenSents: wiki sentences separated into groups with different numbers of tokens
       +-- ...
       +-- 20TokenSents
       +-- 21TokenSents: wiki sentences with 21 bert tokens
          +--textfile
            +-- sentences.txt
       +-- 22TokenSents
       +-- ...
+-- bookcorpus
    +-- TokenSents: wiki sentences separated into groups with different numbers of tokens
      +-- ...
      +-- 20TokenSents
      +-- 21TokenSents: wiki sentences with 21 bert tokens
        +-- textfile
          +-- sentences.txt
      +-- 22TokenSents
      +-- ...
```
