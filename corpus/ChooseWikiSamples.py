import torch
import numpy as np
import pickle
import csv
from multiprocessing import Pool
import os
import argparse
import glob
import time
import matplotlib.pyplot as plt
import math

def ExtractSentNum(folder_id,folder_path):
    with open(f'{folder_path}','r') as f:
        reader = csv.reader(f)
        file = [row for row in reader]
    return len(file)-1

def ExtractSents(folder_path,sent_ids):
    with open(f'{folder_path}','r') as f:
        reader = csv.reader(f)
        file = [row for row in reader]
    return (file[0],folder_path,[file[sent_id+1] for sent_id in sent_ids])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_tokens',type=int, required=True,
                        help='number of tokens including special tokens')
    parser.add_argument('--num_samples',type=int, default=100)
    args = parser.parse_args()
    text_path = f'WikiData/TokenSents/{args.num_tokens}TokenSents/textfile/'
    folder_path_list = glob.glob(f'{text_path}SentProbs/*.csv')
    arg = [(folder_id,folder_path) for folder_id,folder_path in enumerate(folder_path_list)]
    with Pool(processes=50) as p:
        num_sents = p.starmap(ExtractSentNum,arg)
    assert len(folder_path_list)==len(num_sents)
    num_sents = np.array(num_sents)
    print(num_sents)
    
    # Fix the seed for reproducibility
    np.random.seed(seed=2021)
    
    for i in range(args.num_samples):
        sampled_sents = np.random.choice(np.arange(num_sents.sum()),size=1000,replace=False)
        sampled_folders = [(num_sents.cumsum()<=sent_id).sum() for sent_id in sampled_sents]
        dir_dict = {}
        for folder_id,sent_id in zip(sampled_folders,sampled_sents):
            if folder_id not in dir_dict:
                dir_dict[folder_id] = []
            if folder_id==0:
                dir_dict[folder_id].append(sent_id)
            else:
                dir_dict[folder_id].append(sent_id-num_sents.cumsum()[folder_id-1])
        arg = [(folder_path_list[folder_id],sent_ids) for folder_id,sent_ids in dir_dict.items()]
        with Pool(processes=50) as p:
            sent_list = p.starmap(ExtractSents,arg)

        os.makedirs(f'{text_path}SampledSents',exist_ok=True)
        with open(f'{text_path}SampledSents/{i}.csv','w') as f:
            writer = csv.writer(f)
            writer.writerow(['sample_id','folder_id']+sent_list[0][0])
            j=0
            for line in sent_list:
                for row in line[2]:
                    writer.writerow([j,line[1].replace(f'{text_path}SentProbs/','').replace('.csv','')]+row)
                    j+=1
