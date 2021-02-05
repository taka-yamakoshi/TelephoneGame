# TelephoneGame
informal description of the structure of this repo:

Since I cannot track all the data directories with git, I describe the intended directory structure here.

```
.
+-- initial_sentences
|   +-- 10Tokens
|       +-- input.txt
|   +-- 20Tokens
|   +-- 30Tokens
+-- BertData
|   +-- ...
|   +-- 12TokenSents
|   +-- 13TokenSents: bert sentences generated with 13 tokens
|      +-- textfile
|         +-- bert-base-uncased
|            +-- 20_11000: batch_size and chain_len
|               +-- bert_gibbs_~.csv: output files from `bert.py`
|               +-- ...
|               +-- TrackFreq: tracked sentence lengths, POS, TAG
|                  +-- ~.csv: output files from `TrackFreq.py`
|                  +-- ...
|            +-- ...
|         +-- bert-large-cased
|      +-- datafile
|         +-- bert-base-uncased: model name
|            +-- 20_11000_10: batch_size, chain_len, and sent_sample
|               +-- POSFreqAllBert.pkl: output files from `CountFreq.py`
|               +-- ...
|            +-- ...
|         +-- bert-large-cased
|   +-- 14TokenSents
|   +-- 15TokenSents
|   +-- ...
+-- WikiData
    +-- Extracted: raw wiki pages (.json files) extracted using `WikiExtracter.py`
    +-- TokenSents: wiki sentences separated into groups with different numbers of tokens
       +-- ...
       +-- 12TokenSents
       +-- 13TokenSents: wiki sentences with 13 bert tokens
          +-- AA.txt: wiki sentences separeted with '\n'
          +-- AB.txt
          +-- ...
       +-- 14TokenSents
       +-- 15TokenSents
       +-- ...
```
