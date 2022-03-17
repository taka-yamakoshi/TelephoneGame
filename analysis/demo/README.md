## Demo
This directory contains scripts for creating the demo.

First, please use `extract_sample_sentences.py` or a similar script to extract chains and create `sample_sentences.csv`.

Then, `visualize_tsne.py` will read `sample_sentences.csv` and run t-SNE using the sentence embeddings provided by `SentenceTransformer`.

To make a demo video, please use `make_movie.py`.  To run the demo, please use `app.py`.
