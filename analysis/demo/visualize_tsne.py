import pandas as pd

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-distilroberta-base-v1')
chain = pd.read_csv('sample_sentences.csv')
chain = chain.assign(cleaned_sentence=chain.sentence.str.replace(r'\[CLS\] ', '',regex=True).str.replace(r' \[SEP\]', '',regex=True))
sentence_embeddings = model.encode(chain.cleaned_sentence.to_list(), show_progress_bar=False)

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn.decomposition import TruncatedSVD
tsne = TSNE(n_components = 2, n_iter=2000)
big_pca = PCA(n_components = 50)

tsne_vals = tsne.fit_transform(big_pca.fit_transform(sentence_embeddings))
tsne = pd.concat([chain, pd.DataFrame(tsne_vals, columns = ['x_tsne', 'y_tsne'],index=chain.index)], axis = 1)
tsne.to_csv('tsne_out.csv')
