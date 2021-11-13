## Sentence Generation

First, set the environment variable `BERT_PATH` for the generated sentences.
```{python3}
export BERT_PATH='YOUR PATH TO BERT DATA'
```
Then, run `bert.py`; an example set of parameters would be
```{python3}
python bert.py --sentence_id input --core_id 0 --num_tokens 21 --sampling_method gibbs --sweep_order random --batch_size 10 --chain_len 2000
```
*`step` and `sweep`
At each `step`, we sample one position in the sentence.
For each `sweep`, we repeat these steps until all the positions (except the special tokens at the beginning and end) are sampled.
This means each `sweep` is `num_tokens-2` steps.
*`batch_size` is the number of chains for each initial sentence and `chain_len` is the number of sweeps for each chain.
For behavioral experiment, we used `batch_size=10` and `chain_len=2000`.
For corpus comparison, we used `batch_size=5` and `chain_len=51000` with the first 1000 steps as burn-in period.
*`sampling_method` defines the sampling algorithm.  For the pseudo-Gibbs sampling, the options are `gibbs` (original) or `gibbs_mixture` (with mixture kernel).  For the Metropolis-Hastings sampling, the options are `mh_{num_masks}` or `mh_mixture_{num_masks}`, where `num_masks` is the number of tokens to be sampled at the same time.
*`sweep_order` defines the order of positions to be sampled.  `random` chooses random positions with replacement, `random_sweep` chooses random positions without replacement.  We used `random` for pseudo-Gibbs and `random_sweep` for MH.
*`mask_initialization` is to be used when starting chains with mask tokens.

The structure of the output directory would be as follows:
```
+-- BERT_PATH
| +-- ...
| +-- 11TokenSents: bert sentences generated with 11 tokens
|   +-- textfile
|     +-- bert-base-uncased
|       +-- 10_2000: batch_size and chain_len
|         +-- bert_gibbs_~.csv: output files from `bert.py`
|         +-- ...
| +-- 12TokenSents
| +-- 21TokenSents
| +-- ...
```
