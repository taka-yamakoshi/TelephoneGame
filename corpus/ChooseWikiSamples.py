# export WIKI_PATH='YOUR PATH TO WIKICORPUS'
# export BOOK_PATH='YOUR PATH TO BOOKCORPUS'
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

def ExtractSents(folder_path,sent_ids):
    with open(f'{folder_path}','r') as f:
        reader = csv.reader(f)
        file = [row for row in reader]
    return (file[0],folder_path,[file[sent_id+1] for sent_id in sent_ids])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, choices=['wiki','book'])
    parser.add_argument('--num_tokens',type=int, required=True,
                        help='number of tokens including special tokens')
    parser.add_argument('--num_samples',type=int, default=100)
    args = parser.parse_args()
    if args.corpus=='wiki':
        text_path = f'{os.environ.get("WIKI_PATH")}/TokenSents/{args.num_tokens}TokenSents/textfile/'
    elif args.corpus=='book':
        text_path = f'{os.environ.get("BOOK_PATH")}/TokenSents/{args.num_tokens}TokenSents/textfile/'
    with open(f'{text_path}SentProbs/sentences.csv','r') as f:
        reader = csv.reader(f)
        file = [row for row in reader]
    num_sents = len(file)-1
    print(num_sents)

    # Fix the seed for reproducibility
    np.random.seed(seed=2021)

    for i in range(args.num_samples):
        sampled_sent_ids = np.random.choice(np.arange(num_sents),size=1000,replace=False)
        os.makedirs(f'{text_path}SampledSents',exist_ok=True)
        with open(f'{text_path}SampledSents/{i}.csv','w') as f:
            writer = csv.writer(f)
            writer.writerow(['sample_id']+file[0])
            for sample_id,sent_id in enumerate(sampled_sent_ids):
                writer.writerow([sample_id]+file[sent_id+1])
