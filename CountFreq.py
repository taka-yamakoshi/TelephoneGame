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

def LoadCorpus(args,file_name,text_path,nlp,burn_in=True):
    '''
        Load either wikipedia or bert-generated sentences
        For bert, the first 1000 is for the burn-in period
    '''
    if args.corpus == 'wiki':
        with open(f'{text_path}{file_name}.txt','r') as f:
            text = f.read().split('\n')[:-1]
        doc = nlp.pipe(text)
    elif args.corpus == 'bert':
        with open(f'{text_path}{file_name}.csv','r') as f:
            reader = csv.reader(f)
            file = [row for row in reader]
            head = file[0]
            text = file[1:]
        if burn_in:
            Converged = [TakeOutFuncTokens(row[head.index(f'sentence')]) for row in text if int(row[head.index('iter_num')]) > 1000 and int(row[head.index('iter_num')])%args.sent_sample==0]
        else:
            Converged = [TakeOutFuncTokens(row[head.index(f'sentence')]) for row in text if int(row[head.index('iter_num')])%args.sent_sample==0]
        doc = nlp.pipe(Converged)
    return doc

def ExtractFreq(args,file_name,text_path,nlp):
    '''
        Extract frequency of the specified 'metric' and return dictionary
    '''
    doc = LoadCorpus(args,file_name,text_path,nlp)
    Freq = {}
    ShortDep = ""
    sent_num = 0
    for line in doc:
        if len(list(line.sents)) == 1:
            if len(bert_tokenizer(line.text)['input_ids']) == args.num_tokens:
                sent_num += 1
                if args.metric in ['dep','dep_norm']:
                    total_dist = np.array([abs(token_pos-token.head.i) for token_pos,token in enumerate(line)]).sum()
                    if args.metric == 'dep':
                        if total_dist in Freq:
                            Freq[total_dist] += 1
                        else:
                            Freq[total_dist] = 1
                    elif args.metric == 'dep_norm':
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
                        if args.metric in ['vocab', 'pos', 'tag']:
                            if args.metric == 'vocab':
                                word = token.text
                            elif args.metric == 'pos':
                                word = token.pos_
                            elif args.metric == 'tag':
                                word = token.tag_
                            if word in Freq:
                                Freq[word] += 1
                            else:
                                Freq[word] = 1
                        elif args.metric in ['pos_vocab', 'tag_vocab']:
                            if args.metric == 'pos_vocab':
                                word = token.pos_
                            elif args.metric == 'tag_vocab':
                                word = token.tag_
                            if word in Freq:
                                if token.text.lower() in Freq[word]:
                                    Freq[word][token.text.lower()] += 1
                                else:
                                    Freq[word][token.text.lower()] = 1
                            else:
                                Freq[word] = {}
                                Freq[word][token.text.lower()] = 1
    if args.corpus == 'wiki':
        with open(f'{data_path}CountFiles/{args.metric.upper()}Freq{file_name}.pkl','wb') as f:
            pickle.dump(Freq,f)
    elif args.corpus == 'bert':
        with open(f'{data_path}CountFiles/{args.metric.upper()}FreqBert{file_name}.pkl','wb') as f:
            pickle.dump(Freq,f)
    print(f'# of sentences for {file_name}: {sent_num}')
    return [Freq,ShortDep]

if __name__ == '__main__':
    ##Organize arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type = str, required = True)
    parser.add_argument('--metric', type = str, required = True)
    parser.add_argument('--model', type = str, required = True)
    #The rest are only for bert
    parser.add_argument('--batch_size', type = int)
    parser.add_argument('--chain_len', type = int)
    parser.add_argument('--sent_sample', type = int)
    parser.add_argument('--num_tokens', type = int, default = 13)
    args = parser.parse_args()
    assert args.corpus in ['wiki', 'bert'], 'Invalid corpus name'
    assert args.metric in ['vocab', 'pos', 'tag', 'dep', 'pos_vocab', 'tag_vocab', 'dep_norm'], 'Invalid metric name'
    print('running with args', args)
    
    from transformers import BertTokenizer
    bert_tokenizer = BertTokenizer.from_pretrained(args.model)
    
    ##Specify proper paths and gather file names
    if args.corpus == 'bert':
        assert args.chain_len != None and args.batch_size != None and args.sent_sample != None, ''
        text_path = f'BertData/{args.num_tokens}TokenSents/textfile/{args.model}/{args.batch_size}_{args.chain_len}/bert_gibbs_input_'
        data_path = f'BertData/{args.num_tokens}TokenSents/datafile/{args.model}/{args.batch_size}_{args.chain_len}_{args.sent_sample}/'
        files = [file_name.replace(f'{text_path}','').replace('.csv','') for file_name in glob.glob(f'{text_path}*.csv')]
    elif args.corpus == 'wiki':
        text_path = f'WikiData/TokenSents/{args.num_tokens}TokenSents/textfile/'
        data_path = f'WikiData/TokenSents/{args.num_tokens}TokenSents/datafile/'
        os.makedirs(data_path,exist_ok=True)
        files = [file_name.replace(f'{text_path}','').replace('.txt','') for file_name in glob.glob(f'{text_path}*.txt')]

    ##Create output dirs when necessary
    os.makedirs(f'{data_path}CountFiles/',exist_ok=True)
    os.makedirs(f'{data_path}ShortDepSents/',exist_ok=True)

    ##Set up the spacy tokenizer
    nlp = TokenizerSetUp()

    arg = [(args,file_name,text_path,nlp) for file_name in files]
    ##Extract frequency
    with Pool(processes=50) as p:
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
