# Analysis

## RNotebooks and demo

The sub-directory `RNotebooks` contains the R notebooks for running analyses and producing figures.
The sub-directory `demo` contains scripts for creating the demo video and the interactive webpage.

## Computing corpus features

`CountFreq.py` extracts different linguistic features (POS/TAG/DEP etc.) for different corpora.

First, set the environment variables for wikipedia, bookcorpus, or BERT data.

For example, set `WIKI_PATH` to be the path to the wikipedia corpus
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

`metric` is the type of linguistic feature to count.  The options include `pos` (parts of speech), `tag` (detailed parts of speech tag), `dep` (dependency label) and `dep_dist` (dependency distance: the sum of distances between the head and the child for each sentence).  In order to ensure independent samples, we extract samples from chains with lags (please see Sec. 3.2 of the paper for details).  `sent_sample` specifies this lag, and the value we used was 500 (sweeps).

Running `CountFreq.py` will add `datafile` directory inside the corresponding corpus directory.

For example, for the BERT data,
```
└── BERT_PATH
    ├── ...
    ├── 11TokenSents: bert sentences generated with 11 tokens
    │   ├── textfile
    │   └── datafile
    │       └── bert-base-uncased: model name
    │           └── 5_51000_500_gibbs_mixture_random_mask_init_1: paramters for the generated sentences
    │               ├── POSFreqAllBert.pkl: output files from `CountFreq.py`
    │               ├── TAGFreqAllBert.pkl: output files from `CountFreq.py`
    │               └── ...
    ├── 12TokenSents
    ├── 21TokenSents
    └── ...
```

## Extracting linguistic features for sentences used in the behavioral experiment

`ExtractLingFeatures.py` reads in the result of the behavioral experiment and creates a csv file containing linguistic features, which we used when predicting naturalness ratings.
