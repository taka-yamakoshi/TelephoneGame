import pandas as pd

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-distilroberta-base-v1')
chain = (pd.read_csv('../behavioral_experiment/stimuli/bert/12TokenSents.csv')
         .query("chain_num == 0 and sentence_num == 6"))
sentences = chain['sentence'].str.replace(r'\[CLS\] ', '').str.replace(r' \[SEP\]', '').tolist()
sentence_embeddings = model.encode(sentences)

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn.decomposition import TruncatedSVD
tsne = TSNE(n_components = 2, n_iter=2000)
big_pca = PCA(n_components = 50)

tsne_vals = tsne.fit_transform(big_pca.fit_transform(sentence_embeddings))
tsne = pd.concat([chain, pd.DataFrame(tsne_vals, columns = ['x_tsne', 'y_tsne'],index=chain.index)],
                    axis = 1)
tsne.to_csv('tsne_out.csv')
