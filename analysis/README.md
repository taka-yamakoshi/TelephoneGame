## Behavioral Experiment

## Corpus Distribution Comparison
### Count Linguistic Features
First, set the environment variables for wikipedia, bookcorpus, or BERT data.
For example,
```{python3}
export WIKI_PATH='YOUR PATH TO WIKICORPUS'
```
Then, run `CountFreq.py`; an example set of parameters for BERT data would be
```{python3}
python CountFreq.py --corpus bert --num_tokens 11 --metric pos --sent_sample 500 --sampling_method gibbs --batch_size 5 --chain_len 51000 --sweep_order random --mask_initialization
```
or for wikipedia/bookcorpus
```{python3}
python CountFreq.py --corpus wiki --num_tokens 21 --metric $METRIC
```

*`metric` is the type of linguistic feature to count.  The options include `pos` (parts of speech), `tag` (detailed parts of speech tag), `dep` (dependency label) and `dep_dist` (dependency distance: the sum of distances between the head and the child for each sentence).

This will add `datafile` directory inside the corresponding corpus directory.
Example: for BERT data,
```
+-- BERT_PATH
| +-- ...
| +-- 11TokenSents: bert sentences generated with 11 tokens
|   +-- textfile
|   +-- datafile
|     +-- bert-base-uncased: model name
|       +-- 5_51000_500_gibbs_mixture_random_mask_init_1: paramters for the generated sentences
|         +-- POSFreqAllBert.pkl: output files from `CountFreq.py`
|         +-- ...
| +-- 12TokenSents
| +-- 21TokenSents
| +-- ...
```
