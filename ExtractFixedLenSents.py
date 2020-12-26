import torch
import numpy as np
import pickle
import csv
import sys
import spacy
from spacy.symbols import ORTH
import json
import time
from multiprocessing import Pool
import os
import argparse
import glob
sys.path.append('..')

def ExtractSentences(i,folder_path):
    if i < 10:
        file_id = f'0{i}'
    else:
        file_id = f'{i}'
    with open(f'{folder_path}/wiki_{file_id}','r') as infile:
        file = infile.read().split('\n')[:-1]
        sent_list = []
        for page in file:
            text = json.loads(page)['text'].replace('\n',' ')
            #text = text.encode('ascii','replace').decode('ascii')
            line = nlp(text)
            for sent in line.sents:
                if len(sent) == args.num_tokens:
                    sent_list.append(sent.text+'\n')
    return ''.join(sent_list)

def ExtractSentencesBert(i,folder_path):
    if i < 10:
        file_id = f'0{i}'
    else:
        file_id = f'{i}'
    with open(f'{folder_path}/wiki_{file_id}','r') as infile:
        file = infile.read().split('\n')[:-1]
        sent_list = []
        for page in file:
            text = json.loads(page)['text'].replace('\n',' ')
            #text = text.encode('ascii','replace').decode('ascii')
            for sent in nlp(text).sents:
                #Below is to avoid sentences longer than the maximum sequence length for BERT (512 tokens)
                if len(sent)<50:
                    if len(bert_tokenizer(sent.text)['input_ids'])==args.num_tokens:
                        sent_list.append(sent.text)
    return ''.join(sent_list)

def TokenizerSetUp():
    nlp = spacy.load('en_core_web_lg')
    nlp.tokenizer.add_special_case("[UNK]",[{ORTH: "[UNK]"}])
    sentencizer = nlp.create_pipe("sentencizer")
    for punct_char in ['.',':',';','!','?']:
        sentencizer.punct_chars.add(punct_char)
    nlp.add_pipe(sentencizer,first=True)
    return nlp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', type = str, required = True)
    parser.add_argument('--num_tokens', type = int, required = True)
    args = parser.parse_args()
    text_path = 'WikiData/Extracted/'
    folder_path_list = glob.glob(f'{text_path}*')
    nlp = TokenizerSetUp()
    if args.tokenizer.lower() == 'bert':
        from transformers import BertTokenizer
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for folder_path in folder_path_list:
        folder_name = folder_path.replace(text_path,'')
        print(folder_name)
        time1 = time.time()
        files = os.listdir(folder_path)
        arg = [(i,folder_path) for i,file in enumerate(files)]
        if args.tokenizer.lower() == 'bert':
            with Pool(processes=100) as p:
                sentence_list = p.starmap(ExtractSentencesBert,arg)
        elif args.tokenizer.lower() == 'spacy':
            with Pool(processes=100) as p:
                sentence_list = p.starmap(ExtractSentences,arg)
        with open(f'WikiData/10WordSents/textfile/{folder_name}.txt','w') as outfile:
            outfile.write(''.join(sentence_list))
        time2 = time.time()
        print(time2-time1)
