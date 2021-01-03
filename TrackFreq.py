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
import glob
import argparse
from ExtractFixedLenSents import TokenizerSetUp
from CountFreq import LoadCorpus
from transformers import BertTokenizer
import re

def WriteOut(args,file_name,text_path,nlp):
    print(f'Processing {file_name}')
    sent_num = 0
    sent_id = 0
    doc = LoadCorpus(args,file_name,text_path,nlp,burn_in=False)
    head = ['sent_id','sentence','num_tokens','seq_len','POS','TAG','DEP']
    with open(f'textfile/{args.model}/{args.batch_size}_{args.chain_len}/TrackFreq/{file_name}.csv','w') as f:
        writer = csv.writer(f)
        writer.writerow(head)
        for line in doc:
            output, sent_flag = TrackFreq(line)
            writer.writerow([sent_id] + output)
            sent_id += 1
            if sent_flag:
                sent_num += 1
    print(f'Number of sentences for {file_name}: {sent_num}')

def TrackFreq(line):
    sent_flag = False
    seq_len = len(line)
    token_num = len(bert_tokenizer(line.text)['input_ids'])
    if token_num == args.num_tokens:
        sent_flag = True
    '''
    elif len(bert_tokenizer(line.text.replace('...','. '))['input_ids']) == args.num_tokens or len(bert_tokenizer(re.sub(r'\[unused.+\]','.',line.text))['input_ids']) == args.num_tokens:
        sent_flag = True
    '''
    total_dist = np.array([abs(token_pos-token.head.i) for token_pos,token in enumerate(line)]).sum()
    POSList = []
    TAGList = []
    for token in line:
        POSList.append(token.pos_)
        TAGList.append(token.tag_)
    return [line.text,token_num,seq_len,' '.join(POSList),' '.join(TAGList),total_dist],sent_flag

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, required = True)
    parser.add_argument('--batch_size', type = int, required = True)
    parser.add_argument('--chain_len', type = int, required = True)
    parser.add_argument('--sent_sample', type = int, required = True,
                        help='frequency of recording sentences')
    parser.add_argument('--num_tokens', type = int, default = 13)
    args = parser.parse_args()
    print('running with args', args)
    args.corpus = 'bert'
    text_path = f'textfile/{args.model}/{args.batch_size}_{args.chain_len}/bert_gibbs_input_'
    files = [file_name.replace(f'{text_path}','').replace('.csv','') for file_name in glob.glob(f'{text_path}*.csv')]
    nlp = TokenizerSetUp()
    arg = [(args,file_name,text_path,nlp) for file_name in files]
    bert_tokenizer = BertTokenizer.from_pretrained(args.model)
    with Pool(processes=100) as p:
        Results = p.starmap(WriteOut,arg)
