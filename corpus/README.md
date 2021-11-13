## Sentence Extraction

### Wikipedia
1. Download wikipedia corpus
We used preprocessed wikipedia data from [GluonNLP](https://github.com/dmlc/gluon- nlp/tree/master/scripts/datasets/pretrain corpus) .
2. Sentencize and tokenize wikipedia sentences
Set the environment variable `WIKI_PATH` for the wikipedia corpus.
```{python3}
export WIKI_PATH='YOUR PATH TO WIKICORPUS'
```
Run `ExtractFixedLenSents.py`
3. See the distribution of sentence lengths
```{python3}
python CheckSentNums.py --corpus wiki
```
4. Calculate the bert scores for wikipedia sentences (this is likely to take a long time even with GPU)
Example: for 21 token sentences
```{python3}
python CalcSentProb.py --corpus wiki --num_tokens 21
```

 The structure of the directory containing the wikipedia sentences would be as follows:
```
+-- wikicorpus
| +-- prepared_wikipedia: preprocessed wikipedia corpus from GluonNLP
| +-- TokenSents: wiki sentences separated into groups with different numbers of tokens
|   +-- ...
|   +-- 20TokenSents
|   +-- 21TokenSents: wiki sentences with 21 bert tokens
|     +--textfile
|       +-- SentProbs
|         +-- sentences.csv
|       +-- sentences.txt
|   +-- 22TokenSents
|   +-- ...
```

### BookCorpus
1. Download bookcorpus
Install huggingface `datasets`.
When you run `ExtractFixedLenSents.py` in step 2, it will automatically download bookcorpus from `datasets`.
2. Sentencize and tokenize bookcorpus sentences
Set the environment variable `BOOK_PATH` for the wikipedia corpus.
```{python3}
export BOOK_PATH='YOUR PATH TO BOOKCORPUS'
```
Run `ExtractFixedLenSents.py`
3. See the distribution of sentence lengths
```{python3}
python CheckSentNums.py --corpus book
```
4. Calculate the bert scores for wikipedia sentences (this is likely to take a long time even with GPU)
Example: for 21 token sentences
```{python3}
python CalcSentProb.py --corpus book --num_tokens 21
```

The structure of the directory containing the bookcorpus sentences would be as follows:
```
+-- bookcorpus
| +-- TokenSents: bookcorpus sentences separated into groups with different numbers of tokens
|   +-- ...
|   +-- 20TokenSents
|   +-- 21TokenSents: book sentences with 21 bert tokens
|     +--textfile
|       +-- SentProbs
|         +-- sentences.csv
|       +-- sentences.txt
|   +-- 22TokenSents
|   +-- ...
```

### Choosing samples for behavioral experiment
