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

def CalcFreqHalf(id_list):
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

metric = args[1]

alphabet_list = [chr(ord('A')+i) for i in range(26)]
folder_name_list = [char1+char2 for char1 in alphabet_list for char2 in alphabet_list][:141]

random_array = np.random.permutation(140)
A_list = list(random_array[:70]) + [140]
B_list = list(random_array[70:])
assert len(A_list) == 71
assert len(B_list) == 70
assert len(set(A_list+B_list)) == 141

print('Calculating for the first half')
FreqDictHalfA = CalcFreqHalf(A_list)
with open(f'../WikiData/10WordSents/{metric.upper()}FreqHalfA.pkl','wb') as f:
    pickle.dump(FreqDictHalfA,f)

print('Calculating for the second half')
FreqDictHalfB = CalcFreqHalf(B_list)
with open(f'../WikiData/10WordSents/{metric.upper()}FreqHalfB.pkl','wb') as f:
    pickle.dump(FreqDictHalfB,f)
