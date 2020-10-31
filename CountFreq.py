import torch
import numpy as np
import pickle
import csv
import sys
import spacy
from spacy.symbols import ORTH
import time
from multiprocessing import Pool
import os

sys.path.append('..')
args = sys.argv

alphabet_list = [chr(ord('A')+i) for i in range(26)]
folder_name_list = [char1+char2 for char1 in alphabet_list for char2 in alphabet_list][:141]

def TakeOutFuncTokens(sentence):
    return ' '.join(sentence.split(' ')[1:-1])

def LoadCorpus(name,id):
    if name == 'wiki':
        with open(f'../WikiData/10WordSents/{id}.txt','r') as f:
            text = f.read().split('\n')[:-1]
        doc = nlp.pipe(text)
    elif name == 'bert':
        with open(f'textfile/bert_gibbs_input_{id}_1_fixed.csv','r') as f:
            reader = csv.reader(f)
            file = [row for row in reader]
            head = file[0]
            text = file[1:]
        Converged = [TakeOutFuncTokens(row[head.index(f'chain {j}')]) for row in text for j in range(10) if int(row[head.index('iter_num')]) > 1000]
        doc = nlp.pipe(Converged)
    return doc

def ExtractFreq(id,metric,corpus):
    sent_num = 0
    doc = LoadCorpus(corpus,id)
    Freq = {}
    for line in doc:
        if len(list(line.sents)) == 1:
            sent_num += 1
            if metric == 'dep':
                total_dist = np.array([abs(token_pos-token.head.i) for token_pos,token in enumerate(line)]).sum()
                if total_dist in Freq:
                    Freq[total_dist] += 1
                else:
                    Freq[total_dist] = 1
            else:
                for token in line:
                    if metric == 'vocab':
                        word = token.text
                    elif metric == 'pos':
                        word = token.pos_
                    elif metric == 'tag':
                        word = token.tag_
                    if word in Freq:
                        Freq[word] += 1
                    else:
                        Freq[word] = 1
    if corpus == 'wiki':
        with open(f'../WikiData/10WordSents/CountFiles/{metric.upper()}Freq{id}.pkl','wb') as f:
            pickle.dump(Freq,f)
    elif corpus == 'bert':
        with open(f'datafile/{metric.upper()}FreqBert{id}.pkl','wb') as f:
            pickle.dump(Freq,f)
    print(f'Number of sentences for {id}: {sent_num}')
    return Freq

nlp = spacy.load('en_core_web_lg')
nlp.tokenizer.add_special_case("[UNK]",[{ORTH: "[UNK]"}])
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer,first=True)

corpus = args[1]
metric = args[2]
assert corpus in ['wiki', 'bert'], 'Invalid corpus name'
assert metric in ['vocab', 'pos', 'tag', 'dep'], 'Invalid metric name'

if corpus == 'wiki':
    arg = [(folder_name,metric,corpus) for folder_name in folder_name_list]
elif corpus == 'bert':
    arg = [(i,metric,corpus) for i in range(4)]

with Pool(processes=100) as p:
    DictList = p.starmap(ExtractFreq,arg)

FreqDictAll = {}
for Dict in DictList:
    for word in Dict:
        if word in FreqDictAll:
            FreqDictAll[word] += Dict[word]
        else:
            FreqDictAll[word] = Dict[word]
if corpus == 'wiki':
    with open(f'../WikiData/10WordSents/{metric.upper()}FreqAll.pkl','wb') as f:
        pickle.dump(FreqDictAll,f)
elif corpus == 'bert':
    with open(f'datafile/{metric.upper()}FreqAllBert.pkl','wb') as f:
        pickle.dump(FreqDictAll,f)
