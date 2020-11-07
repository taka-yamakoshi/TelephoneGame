import torch
import numpy as np
import pickle
import csv
import sys
import spacy
import time
from multiprocessing import Pool
import os

sys.path.append('..')
args = sys.argv

def CalcFreqHalf(id_list,metric):
    FreqDictHalf = {}
    for id in id_list:
        with open(f'../WikiData/10WordSents/CountFiles/{metric.upper()}Freq{folder_name_list[id]}.pkl','rb') as f:
            Dict = pickle.load(f)
        for word in Dict:
            if word in FreqDictHalf:
                FreqDictHalf[word] += Dict[word]
            else:
                FreqDictHalf[word] = Dict[word]
    return FreqDictHalf

alphabet_list = [chr(ord('A')+i) for i in range(26)]
folder_name_list = [char1+char2 for char1 in alphabet_list for char2 in alphabet_list][:141]

def random_split(split_iter,metric):
    np.random.seed(split_iter)
    random_list = np.arange(len(folder_name_list))[np.random.rand(len(folder_name_list)) > 0.5]
    print(f'Files included in split number {split_iter}: {len(random_list)}')
    FreqDictHalf = CalcFreqHalf(random_list,metric)
    return FreqDictHalf

metric = args[1]
split_num = 100
arg = [(i,metric) for i in range(split_num)]
with Pool(processes=100) as p:
    DictList = p.starmap(random_split,arg)

with open(f'../WikiData/10WordSents/{metric.upper()}FreqHalf.pkl','wb') as f:
    pickle.dump(DictList,f)
