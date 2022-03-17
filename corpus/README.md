## Sentence Extraction

### Wikipedia
1. Download wikipedia corpus  
We used preprocessed wikipedia data from [GluonNLP](https://github.com/dmlc/gluon-nlp/tree/master/scripts/datasets/pretrain_corpus).

2. Sentencize and tokenize wikipedia sentences  
First, please set the environment variable `WIKI_PATH` to be the path to the downloaded Wikipedia corpus.
  ```{python3}
  export WIKI_PATH='YOUR PATH TO WIKICORPUS'
  ```
Then, please run
  ```{python3}
  python ExtractFixedLenSents.py --corpus wiki
  ```
To see the distribution of sentence lengths, please run
  ```{python3}
  python CheckSentNums.py --corpus wiki
  ```

3. Calculate the "BERT scores" for wikipedia sentences (this is likely to take a long time even with GPU)  
Example: for sentences with 21 tokens
  ```{python3}
  python CalcSentProb.py --corpus wiki --num_tokens 21
  ```
After running the script, the structure of the directory containing the wikipedia sentences will be as follows:
  ```
  └── wikicorpus
      ├── prepared_wikipedia: preprocessed wikipedia corpus from GluonNLP
      └── TokenSents: wiki sentences separated into groups with different numbers of tokens
          ├── ...
          └── 21TokenSents: wiki sentences with 21 tokens (tokenized by BertTokenizer)
              ├── SentProbs
              │   └── sentences.csv
              └── sentences.txt
  ```

### BookCorpus
1. Download bookcorpus  
Install huggingface `datasets`.  
When you run `ExtractFixedLenSents.py` in step 2, it will automatically download bookcorpus from `datasets`.

2. Sentencize and tokenize bookcorpus sentences  
First, please set the environment variable `BOOK_PATH` to be the path for the bookcorpus.
  ```{python3}
  export BOOK_PATH='YOUR PATH TO BOOKCORPUS'
  ```
Then, please run
  ```{python3}
  python ExtractFixedLenSents.py --corpus book
  ```
To see the distribution of sentence lengths, please run
  ```{python3}
  python CheckSentNums.py --corpus book
  ```

3. Calculate the "BERT scores" for wikipedia sentences (this is likely to take a long time even with GPU)  
Example: for sentences with 21 tokens
  ```{python3}
  python CalcSentProb.py --corpus book --num_tokens 21
  ```
After running the script, the structure of the directory containing the bookcorpus sentences will be as follows:
  ```
  └── bookcorpus
    └── TokenSents: book sentences separated into groups with different numbers of tokens
        └── 21TokenSents: book sentences with 21 tokens (tokenized by BertTokenizer)
            ├── SentProbs
            │   └── sentences.csv
            └── sentences.txt
  ```

## Choosing samples for behavioral experiment
1. First, please set the environment variable `WIKI_PATH` and `BOOK_PATH`.
  ```{python3}
  export WIKI_PATH='YOUR PATH TO WIKICORPUS'
  export BOOK_PATH='YOUR PATH TO BOOKCORPUS'
  ```

2. Sample sentences from corpora.  
`ChooseSamples.py` samples 100 batches of 1000 sentences each.  
Example: for sentences with 21 tokens in Wikipedia
  ```{python3}
  python ChooseSamples.py --corpus wiki --num_tokens 21
  ```

3. Compare the "BERT score" distribution of the samples vs that of the entire corpus.  
`CompareSamples.py` compares the histograms of the "BERT score" for sampled batches and the entire corpus, and reports the batch with a distribution closest to that of the entire corpus.
  ```{python3}
  python CompareSamples.py --corpus wiki --num_tokens 21
  ```
