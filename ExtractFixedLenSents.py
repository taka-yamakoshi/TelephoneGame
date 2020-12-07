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
import unicodedata
import os
import argparse

sys.path.append('..')

def ExtractSentences(i,folder_name):
    if i < 10:
        file_id = f'0{i}'
    else:
        file_id = f'{i}'
    with open(f'../WikiData/Extracted/{folder_name}/wiki_{file_id}','r') as infile:
        file = infile.read().split('\n')[:-1]
        sent_list = []
        for page in file:
            text = json.loads(page)['text'].replace('\n',' ')
            #text = text.encode('ascii','replace').decode('ascii')
            line = nlp(text)
            for sent in line.sents:
                if len(sent) == 11:
                    sent_list.append(sent.text+'\n')
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
    alphabet_list = [chr(ord('A')+i) for i in range(26)]
    folder_name_list = [char1+char2 for char1 in alphabet_list for char2 in alphabet_list][:141]
    nlp = TokenizerSetUp()

    for folder_name in folder_name_list:
        print(folder_name)
        time1 = time.time()
        path = f'../WikiData/Extracted/{folder_name}'
        files = os.listdir(path)
        arg = [(i,folder_name) for i,file in enumerate(files)]
        with Pool(processes=100) as p:
            sentence_list = p.starmap(ExtractSentences,arg)

        with open(f'../WikiData/10WordSents/{folder_name}.txt','w') as outfile:
            outfile.write(''.join(sentence_list))

        time2 = time.time()
        print(time2-time1)
