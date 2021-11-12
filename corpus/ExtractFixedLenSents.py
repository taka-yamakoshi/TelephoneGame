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

def ExtractSentencesOld(i,folder_path):
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
            sentences = [sent.text.strip() for sent in nlp(text).sents]
            sent_list.extend([doc.text+'\n' for doc in nlp.pipe(sentences) if len(doc)==args.num_tokens])
    return ''.join(sent_list)

def ExtractSentencesBertOld(i,folder_path):
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
                if len(sent)<20:
                    sentence = sent.text.strip()
                    if len(bert_tokenizer(sentence)['input_ids'])==args.num_tokens:
                        sent_list.append(sentence+'\n')
    return ''.join(sent_list)

def ExtractSentencesBert(i,folder_path):
    if i < 10:
        file_id = f'0{i}'
    else:
        file_id = f'{i}'
    with open(f'{folder_path}/wiki_{file_id}','r') as infile:
        file = infile.read().split('\n')[:-1]
        sent_dict = {}
        for page in file:
            text = json.loads(page)['text'].replace('\n',' ')
            for sent in nlp(text).sents:
                sentence = sent.text.strip()
                num_tokens = len(bert_tokenizer(sentence)['input_ids'])
                if num_tokens not in sent_dict:
                    sent_dict[num_tokens] = ''
                sent_dict[num_tokens] += sentence+'\n'
    return sent_dict

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
    #parser.add_argument('--num_tokens', type = int, required = True)
    parser.add_argument('--model', type = str)
    args = parser.parse_args()
    text_path = 'WikiData/Extracted/'
    folder_path_list = glob.glob(f'{text_path}*')
    nlp = TokenizerSetUp()
    if args.tokenizer.lower() == 'bert':
        from transformers import BertTokenizer
        bert_tokenizer = BertTokenizer.from_pretrained(args.model)
    for folder_path in folder_path_list:
        folder_name = folder_path.replace(text_path,'')
        print(folder_name)
        time1 = time.time()
        files = os.listdir(folder_path)
        arg = [(i,folder_path) for i,file in enumerate(files)]
        if args.tokenizer.lower() == 'bert':
            with Pool(processes=100) as p:
                sent_dict_list = p.starmap(ExtractSentencesBert,arg)
        elif args.tokenizer.lower() == 'spacy':
            with Pool(processes=100) as p:
                sent_dict_list = p.starmap(ExtractSentences,arg)

        sent_dict_all = {}
        for sent_dict in sent_dict_list:
            for num_tokens,sents in sent_dict.items():
                if num_tokens not in sent_dict_all:
                    sent_dict_all[num_tokens] = ''
                sent_dict_all[num_tokens] += sents

        for num_tokens,sents in sent_dict_all.items():
            os.makedirs(f'WikiData/TokenSents/{num_tokens}TokenSents/textfile/',exist_ok=True)
            with open(f'WikiData/TokenSents/{num_tokens}TokenSents/textfile/{folder_name}.txt','w') as outfile:
                outfile.write(sents)
        time2 = time.time()
        print(time2-time1)
