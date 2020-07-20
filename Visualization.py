import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import csv
import spacy
from sklearn.manifold import TSNE
import sys
sys.path.append('..')
args = sys.argv


def SentEmbedding(sentence):
    words = sentence.split()[1:-1]
    sentence = " ".join(words)
    doc = nlp(sentence)
    contentful = nlp(" ".join([token.lemma_ for token in doc if token.has_vector and token.is_alpha and not (token.is_space or token.is_punct)]))
    return contentful.vector
nlp = spacy.load('en_core_web_lg')

sent_id = 0
temp = 0.8
batch_size = 50
batch_num = 5
chain_len = 500
prob_sample = 25
gen_num = chain_len//prob_sample
with open(f'textfile/bert_{sent_id}_{temp}.csv','r') as f:
    reader = csv.reader(f)
    file = [row for row in reader]
head = file[0]
text = file[1:]
sents = [[[text[chain_len*batch_id+prob_sample*j][head.index(f'chain {chain_num}')] for chain_num in range(batch_size)] for j in range(gen_num)] for batch_id in range(batch_num)]

vecs = np.array([[[SentEmbedding(sent) for sent in generation] for generation in batch] for batch in sents])
assert (batch_num,gen_num,batch_size) == (vecs.shape[0],vecs.shape[1],vecs.shape[2])
plot_vecs = np.empty((batch_num*gen_num*batch_size,vecs.shape[-1]))

for i in range(batch_num):
    for j in range(gen_num):
        for k in range(batch_size):
            plot_vecs[batch_size*gen_num*i+batch_size*j+k] = vecs[i][j][k]
model = TSNE()
embedded = model.fit_transform(plot_vecs)
result = np.empty((batch_num,gen_num,batch_size,2))
for i in range(batch_num):
    for j in range(gen_num):
        for k in range(batch_size):
            result[i][j][k] = embedded[batch_size*gen_num*i+batch_size*j+k]
with open(f'datafile/tsne_{sent_id}_{temp}.pkl','wb') as f:
    pickle.dump(result,f)
