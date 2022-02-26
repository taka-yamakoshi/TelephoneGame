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
import sys
sys.path.append('..')
from analysis.CountFreq import ExtractFreqNew, TokenizerSetUpNew
import itertools
import seaborn as sns

def ExtractSentProbs(folder_id,folder_path):
    with open(f'{folder_path}','r') as f:
        reader = csv.reader(f)
        file = [row for row in reader]
    head = file[0]
    text = file[1:]
    probs = [float(row[head.index('prob_1')]) for row in text]
    return probs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, choices=['wiki','book'])
    parser.add_argument('--num_tokens',type=int, required = True,
                        help='number of tokens including special tokens')
    parser.add_argument('--model',type=str, default='bert-base-uncased')
    args = parser.parse_args()
    if args.corpus=='wiki':
        text_path = f'{os.environ.get("WIKI_PATH")}/TokenSents/{args.num_tokens}TokenSents/textfile/'
        data_path = f'{os.environ.get("WIKI_PATH")}/TokenSents/{args.num_tokens}TokenSents/datafile/'
    elif args.corpus=='book':
        text_path = f'{os.environ.get("BOOK_PATH")}/TokenSents/{args.num_tokens}TokenSents/textfile/'
        data_path = f'{os.environ.get("BOOK_PATH")}/TokenSents/{args.num_tokens}TokenSents/datafile/'

    # Extract sent_prob from all Wikipedia sentences with num_tokens=args.num_tokens
    with open(f'{text_path}SentProbs/sentences.csv','r') as f:
        reader = csv.reader(f)
        file = [row for row in reader]
        head = file[0]
        text = file[1:]
        prob_all = np.array([float(row[head.index('prob_1')]) for row in text])
    print(prob_all.shape)
    print(prob_all[:10])

    # Extract sent_prob from the sampled sentences
    folder_path_list = glob.glob(f'{text_path}SampledSents/*.csv')
    arg = [(folder_id, folder_path) for folder_id,folder_path in enumerate(folder_path_list)]
    with Pool(processes=50) as p:
        prob_sample = p.starmap(ExtractSentProbs,arg)
    color_list = sns.color_palette('Set2')

    # Compare sent_prob
    binned_prob_all,_ = np.histogram(prob_all,bins=np.arange(math.floor(np.min(prob_all)),0),density=True)
    binned_prob_sample = []
    for probs in prob_sample:
        hist,_ = np.histogram(probs,bins=np.arange(math.floor(np.min(prob_all)),0),density=True)
        assert len(hist)==len(binned_prob_all)
        binned_prob_sample.append(hist)
    diff = np.array([((binned_prob-binned_prob_all)**2).sum() for binned_prob in binned_prob_sample])
    print(diff)
    print(diff.argmin())
    print(f'Closest sample: {folder_path_list[diff.argmin()]}')

    # Plot the distributions
    fig = plt.figure(dpi=150,figsize=(10,10))
    for probs in prob_sample:
        probs = np.array(probs)
        plt.hist(probs,
                 bins=np.arange(math.floor(np.min(prob_all)),0),
                 density=True,histtype='step',color=color_list[0],alpha=0.2)
    plt.hist(prob_all,
             bins=np.arange(math.floor(np.min(prob_all)),0),
             density=True,histtype='step',label='all',color=color_list[1],linewidth=2)
    plt.hist(prob_sample[diff.argmin()],
             bins=np.arange(math.floor(np.min(prob_all)),0),
             density=True,histtype='step',label='closest',color=color_list[2],linewidth=2)
    plt.title(f'sent_prob distribution for {args.num_tokens} token sentences\nChosen sample: {folder_path_list[diff.argmin()].replace(f"{text_path}SampledSents/","")}')
    plt.xlabel('sentence log likelihood')
    plt.legend()
    fig.savefig(f'figures/sample_comparison_{args.corpus}_{args.num_tokens}_sent_prob.png')
