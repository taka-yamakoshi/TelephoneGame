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
    return sentence.replace('[CLS]','').replace('[SEP]','').strip()

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
    ShortDep = ""
    for line in doc:
        if len(list(line.sents)) == 1:
            sent_num += 1
            if metric in ['dep','dep_norm']:
                total_dist = np.array([abs(token_pos-token.head.i) for token_pos,token in enumerate(line)]).sum()
                if metric == 'dep':
                    if total_dist in Freq:
                        Freq[total_dist] += 1
                    else:
                        Freq[total_dist] = 1
                elif metric == 'dep_norm':
                    seq_len = len(line)
                    if seq_len in Freq:
                        if total_dist in Freq[seq_len]:
                            Freq[seq_len][total_dist] += 1
                        else:
                            Freq[seq_len][total_dist] = 1
                    else:
                        Freq[seq_len] = {}
                        Freq[seq_len][total_dist] = 1
                if total_dist == 10:
                    ShortDep += line.text+'\n'
            else:
                for token in line:
                    if metric in ['vocab', 'pos', 'tag']:
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
                    elif metric in ['pos_vocab', 'tag_vocab']:
                        if metric == 'pos_vocab':
                            word = token.pos_
                        elif metric == 'tag_vocab':
                            word = token.tag_
                        if word in Freq:
                            if token.text in Freq[word]:
                                Freq[word][token.text] += 1
                            else:
                                Freq[word][token.text] = 1
                        else:
                            Freq[word] = {}
                            Freq[word][token.text] = 1
    if corpus == 'wiki':
        with open(f'../WikiData/10WordSents/CountFiles/{metric.upper()}Freq{id}.pkl','wb') as f:
            pickle.dump(Freq,f)
    elif corpus == 'bert':
        with open(f'datafile/{metric.upper()}FreqBert{id}.pkl','wb') as f:
            pickle.dump(Freq,f)
    print(f'Number of sentences for {id}: {sent_num}')
    return [Freq,ShortDep]

nlp = spacy.load('en_core_web_lg')
nlp.tokenizer.add_special_case("[UNK]",[{ORTH: "[UNK]"}])
sentencizer = nlp.create_pipe("sentencizer")
for punct_char in ['.',',',':',';','!','?']:
    sentencizer.punct_chars.add(punct_char)
nlp.add_pipe(sentencizer,first=True)

corpus = args[1]
metric = args[2]
assert corpus in ['wiki', 'bert'], 'Invalid corpus name'
assert metric in ['vocab', 'pos', 'tag', 'dep', 'pos_vocab', 'tag_vocab', 'dep_norm'], 'Invalid metric name'

if corpus == 'wiki':
    arg = [(folder_name,metric,corpus) for folder_name in folder_name_list]
elif corpus == 'bert':
    arg = [(i,metric,corpus) for i in range(4)]

with Pool(processes=100) as p:
    Results = p.starmap(ExtractFreq,arg)

DictList = []
ShortDepSent = []
for line in Results:
    DictList.append(line[0])
    ShortDepSent.append(line[1])

FreqDictAll = {}
if metric in ['vocab', 'pos', 'tag', 'dep']:
    for Dict in DictList:
        for word in Dict:
            if word in FreqDictAll:
                FreqDictAll[word] += Dict[word]
            else:
                FreqDictAll[word] = Dict[word]
elif metric in ['pos_vocab', 'tag_vocab', 'dep_norm']:
    for Dict in DictList:
        for word in Dict:
            for token_text in Dict[word]:
                if word in FreqDictAll:
                    if token_text in FreqDictAll[word]:
                        FreqDictAll[word][token_text] += Dict[word][token_text]
                    else:
                        FreqDictAll[word][token_text] = Dict[word][token_text]
                else:
                    FreqDictAll[word] = {}
                    FreqDictAll[word][token_text] = Dict[word][token_text]
if corpus == 'wiki':
    with open(f'../WikiData/10WordSents/{metric.upper()}FreqAll.pkl','wb') as f:
        pickle.dump(FreqDictAll,f)
    if metric in ['dep', 'dep_norm']:
        with open('../WikiData/10WordSents/ShortDepSents.txt','w') as f:
            for sentence in ShortDepSent:
                f.write(sentence)
elif corpus == 'bert':
    with open(f'datafile/{metric.upper()}FreqAllBert.pkl','wb') as f:
        pickle.dump(FreqDictAll,f)
    if metric in ['dep', 'dep_norm']:
        with open('datafile/ShortDepSentsBert.txt','w') as f:
            for sentence in ShortDepSent:
                f.write(sentence)
