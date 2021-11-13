# export WIKI_PATH='YOUR PATH TO WIKICORPUS'
# export BOOK_PATH='YOUR PATH TO BOOKCORPUS'
import torch
import numpy as np
import pickle
import csv
import sys
import json
import time
from multiprocessing import Pool
import os
import argparse
import glob
import math

def ExtractSentencesBertNLTK(i,file_name):
    print(f'processing {file_name}')
    start = time.time()
    with open(f'{file_name}','r') as infile:
        file = infile.read().split('\n')
    sent_dict = {}
    for text in file:
        if len(text)>0:
            for sent in sent_tokenize(text.strip()):
                sentence = sent.strip()
                num_tokens = len(bert_tokenizer(sentence)['input_ids'])
                if num_tokens not in sent_dict:
                    sent_dict[num_tokens] = ''
                sent_dict[num_tokens] += sentence+'\n'
    print(f'Time it took for {file_name}: {time.time()-start}')
    return sent_dict

def ExtractSentencesBook(i,text):
    sent_dict = {}
    for line in text:
        for sent in sent_tokenize(line.strip()):
            sentence = sent.strip()
            num_tokens = len(bert_tokenizer(sentence)['input_ids'])
            if num_tokens not in sent_dict:
                sent_dict[num_tokens] = ''
            sent_dict[num_tokens] += sentence+'\n'
    return sent_dict

def batchify(data,batch_size):
    batch_num = math.ceil(len(data)/batch_size)
    return [data[batch_size*i:batch_size*(i+1)] for i in range(batch_num)]

def aggregate_dict_list(sent_dict_list):
    sent_dict_all = {}
    for sent_dict in sent_dict_list:
        for num_tokens,sents in sent_dict.items():
            if num_tokens not in sent_dict_all:
                sent_dict_all[num_tokens] = ''
            sent_dict_all[num_tokens] += sents
    return sent_dict_all

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bert-base-uncased')
    parser.add_argument('--corpus', type=str, required=True, choices=['wiki','book'])
    args = parser.parse_args()

    from transformers import BertTokenizer
    bert_tokenizer = BertTokenizer.from_pretrained(args.model)

    from nltk.tokenize import sent_tokenize
    if args.corpus=='wiki':
        text_path = f'{os.environ.get("WIKI_PATH")}/prepared_wikipedia/'
        files = glob.glob(f'{text_path}*')
        arg = [(i,file) for i,file in enumerate(files)]
        with Pool(processes=100) as p:
            sent_dict_list = p.starmap(ExtractSentencesBertNLTK,arg)

        sent_dict_all = aggregate_dict_list(sent_dict_list)

        for num_tokens,sents in sent_dict_all.items():
            os.makedirs(f'{os.environ.get("WIKI_PATH")}/TokenSents/{num_tokens}TokenSents/textfile/',exist_ok=True)
            with open(f'{os.environ.get("WIKI_PATH")}/TokenSents/{num_tokens}TokenSents/textfile/sentences.txt','w') as outfile:
                outfile.write(sents)
    elif args.corpus=='book':
        from datasets import load_dataset
        data = load_dataset('bookcorpus')
        batched_sentences = batchify(data['train']['text'],1000)
        arg = [(i,sentences) for i,sentences in enumerate(batched_sentences)]
        with Pool(processes=200) as p:
            sent_dict_list = p.starmap(ExtractSentencesBook,arg)

        sent_dict_all = aggregate_dict_list(sent_dict_list)

        for num_tokens,sents in sent_dict_all.items():
            os.makedirs(f'{os.environ.get("BOOK_PATH")}/TokenSents/{num_tokens}TokenSents/textfile/',exist_ok=True)
            with open(f'{os.environ.get("BOOK_PATH")}/TokenSents/{num_tokens}TokenSents/textfile/sentences.txt','w') as outfile:
                outfile.write(sents)
