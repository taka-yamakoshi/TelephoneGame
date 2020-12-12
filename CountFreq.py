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
import argparse
import glob
from ExtractFixedLenSents import TokenizerSetUp

sys.path.append('..')

def TakeOutFuncTokens(sentence):
    '''
        Take out special tokens
    '''
    return sentence.replace('[CLS]','').replace('[SEP]','').strip()

def LoadCorpus(corpus,file_name):
    '''
        Load either wikipedia or bert-generated sentences
        For bert, the first 1000 is for the burn-in period
    '''
    if corpus == 'wiki':
        with open(f'{text_path}{file_name}.txt','r') as f:
            text = f.read().split('\n')[:-1]
        doc = nlp.pipe(text)
    elif corpus == 'bert':
        with open(f'{text_path}{file_name}.csv','r') as f:
            reader = csv.reader(f)
            file = [row for row in reader]
            head = file[0]
            text = file[1:]
        Converged = [TakeOutFuncTokens(row[head.index(f'chain {j}')]) for row in text for j in range(args.batch_size) if int(row[head.index('iter_num')]) > 1000 and int(row[head.index('iter_num')])%args.sent_sample==0]
        doc = nlp.pipe(Converged)
    return doc

def ExtractFreq(file_name,metric,corpus):
    '''
        Extract frequency of the specified 'metric' and return dictionary
    '''
    sent_num = 0
    doc = LoadCorpus(corpus,file_name)
    Freq = {}
    ShortDep = ""
    for line in doc:
        if len(list(line.sents)) == 1:
            if len(line) == 11:
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
                                if token.text.lower() in Freq[word]:
                                    Freq[word][token.text.lower()] += 1
                                else:
                                    Freq[word][token.text.lower()] = 1
                            else:
                                Freq[word] = {}
                                Freq[word][token.text.lower()] = 1
    if corpus == 'wiki':
        with open(f'{data_path}CountFiles/{args.metric.upper()}Freq{file_name}.pkl','wb') as f:
            pickle.dump(Freq,f)
    elif corpus == 'bert':
        with open(f'{data_path}CountFiles/{args.metric.upper()}FreqBert{file_name}.pkl','wb') as f:
            pickle.dump(Freq,f)
    print(f'Number of sentences for {file_name}: {sent_num}')
    return [Freq,ShortDep]

##Organize arguments
parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type = str, required = True)
parser.add_argument('--metric', type = str, required = True)
#The rest are only for bert
parser.add_argument('--model', type = str)
parser.add_argument('--batch_size', type = int)
parser.add_argument('--chain_len', type = int)
parser.add_argument('--sent_sample', type = int)
args = parser.parse_args()
assert args.corpus in ['wiki', 'bert'], 'Invalid corpus name'
assert args.metric in ['vocab', 'pos', 'tag', 'dep', 'pos_vocab', 'tag_vocab', 'dep_norm'], 'Invalid metric name'
print('running with args', args)

##Specify proper paths and gather file names
if args.corpus == 'bert':
    assert args.chain_len != None and args.batch_size != None and args.sent_sample != None
    text_path = f'textfile/{args.model}/{args.batch_size}_{args.chain_len}/bert_gibbs_input_'
    data_path = f'datafile/{args.model}/{args.batch_size}_{args.chain_len}_{args.sent_sample}/'
    files = [file_name.replace(f'{text_path}','').replace('.csv','') for file_name in glob.glob(f'{text_path}*.csv')]
elif args.corpus == 'wiki':
    text_path = 'WikiData/10WordSents/textfile/'
    data_path = 'WikiData/10WordSents/datafile/'
    files = [file_name.replace(f'{text_path}','').replace('.txt','') for file_name in glob.glob(f'{text_path}*.txt')]
arg = [(file_name,args.metric,args.corpus) for file_name in files]

##Set up the spacy tokenizer
nlp = TokenizerSetUp()

##Extract frequency
with Pool(processes=100) as p:
    Results = p.starmap(ExtractFreq,arg)

##Unify the paralleled dictionary outputs to a single dictionary
DictList = []
ShortDepSent = []
for line in Results:
    DictList.append(line[0])
    ShortDepSent.append(line[1])

FreqDictAll = {}
if args.metric in ['vocab', 'pos', 'tag', 'dep']:
    for Dict in DictList:
        for word in Dict:
            if word in FreqDictAll:
                FreqDictAll[word] += Dict[word]
            else:
                FreqDictAll[word] = Dict[word]
elif args.metric in ['pos_vocab', 'tag_vocab', 'dep_norm']:
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

##Write out
if args.corpus == 'wiki':
    with open(f'{data_path}{args.metric.upper()}FreqAll.pkl','wb') as f:
        pickle.dump(FreqDictAll,f)
    if args.metric in ['dep', 'dep_norm']:
        with open(f'{data_path}ShortDepSents/ShortDepSents.txt','w') as f:
            for sentence in ShortDepSent:
                f.write(sentence)
elif args.corpus == 'bert':
    with open(f'{data_path}{args.metric.upper()}FreqAllBert.pkl','wb') as f:
        pickle.dump(FreqDictAll,f)
    if args.metric in ['dep', 'dep_norm']:
        with open(f'{data_path}ShortDepSents/ShortDepSentsBert.txt','w') as f:
            for sentence in ShortDepSent:
                f.write(sentence)
