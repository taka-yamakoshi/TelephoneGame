## Sentence Generation

First, set the environment variable `BERT_PATH` for the generated sentences.
```{python3}
export BERT_PATH='YOUR PATH TO BERT_DATA'
```
Then, run `bert.py`; an example set of parameters would be
```{python3}
python bert.py --sentence_id input --core_id 1 --num_tokens 21 --sampling_method gibbs_mixture --epsilon 1e-3 --sweep_order random
```
