# Analysis

This directory contains several R Notebooks and scripts to reproduce our analysis workflow.

* `chain_analytics.Rmd` checks convergence and auto-correlation of the Gibbs & MH chains
* `corpus_comparison.Rmd` examines the extent to which samples generated from BERT match statistics of its training corpus
* `behavioral_experiment.Rmd` analyses the sentence judgements provided by human participants
  * `gen_stims.Rmd` was used to select balanced sets of sentences to show to participants 

## Computing corpus features

Our corpus distribution analyses rely on computing different metrics on different corpora. These are extracted by `CountFreq.py`.

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

`metric` is the type of linguistic feature to count.  The options include `pos` (parts of speech), `tag` (detailed parts of speech tag), `dep` (dependency label) and `dep_dist` (dependency distance: the sum of distances between the head and the child for each sentence).

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
