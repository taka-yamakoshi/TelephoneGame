import numpy as np
import pickle
import csv
import sys
import time
from multiprocessing import Pool
import os
import glob
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

def ExtractTrackFreq(args,file_name,path):
    with open(f'{path}{file_name}.csv','r') as f:
        reader = csv.reader(f)
        file = [row for row in reader]
    head = file[0]
    text = file[1:]
    assert len(text) == args.num_chain_per_file*args.extracted_chain_len*args.batch_size
    if args.metric == 'spacy_tokens':
        FreqCount = [[[int(text[chain_num*args.extracted_chain_len*args.batch_size+args.batch_size*iter+batch_num][head.index('spacy_tokens')]) for iter in range(args.extracted_chain_len)] for batch_num in range(args.batch_size)] for chain_num in range(args.num_chain_per_file)]
        return FreqCount
    elif args.metric == 'prob':
        FreqCount = [[[float(text[chain_num*args.extracted_chain_len*args.batch_size+args.batch_size*iter+batch_num][head.index('prob')]) for iter in range(args.extracted_chain_len)] for batch_num in range(args.batch_size)] for chain_num in range(args.num_chain_per_file)]
        return FreqCount
    elif args.metric in ['pos','tag']:
        FreqCount = [[[int(text[chain_num*args.extracted_chain_len*args.batch_size+args.batch_size*iter+batch_num][head.index(args.metric.upper())]) for iter in range(args.extracted_chain_len)] for batch_num in range(args.batch_size)] for chain_num in range(args.num_chain_per_file)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str)
    parser.add_argument('--metric', type = str)
    parser.add_argument('--batch_size', type = int)
    parser.add_argument('--chain_len', type = int)
    parser.add_argument('--sent_sample', type = int)
    args = parser.parse_args()
    print('running with args', args)
    path = f'textfile/{args.model}/{args.batch_size}_{args.chain_len}/TrackFreq/'
    files = [file_name.replace(f'{path}','').replace('.csv','') for file_name in glob.glob(f'{path}*.csv')]
    
    func_args = [(args,file_name,path) for file_name in files]
    num_init_sents = 40
    num_chain_per_file = num_init_sents//len(func_args)
    extracted_chain_len = args.chain_len//args.sent_sample
    args.num_chain_per_file = num_chain_per_file
    args.extracted_chain_len = extracted_chain_len
    print(f'# of initial sentences per file: {num_chain_per_file}')
    print(f'# of sentences per chain: {extracted_chain_len}')

    with Pool(processes=100) as p:
        FreqCountList = p.starmap(ExtractTrackFreq,func_args)
    FreqCountArray = np.array(FreqCountList).reshape((num_init_sents,len(FreqCountList[0][0]),len(FreqCountList[0][0][0])))

    color_list = sns.color_palette('Set2')
    color_seq = sns.color_palette("coolwarm",n_colors=40)
    fig = plt.figure(figsize=(15,5),dpi=200)
    for chain_num,FreqCount in enumerate(FreqCountArray):
        #plt.plot(FreqCount.T,alpha=0.01, color=color_seq[chain_num])
        plt.plot(np.average(FreqCount,axis=0), color=color_seq[chain_num])
    fig.savefig(f'figures/TrackFreq_{args.model}_{args.batch_size}_{args.chain_len}_{args.sent_sample}_{args.metric}.png')
