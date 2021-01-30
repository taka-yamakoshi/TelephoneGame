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
from PlotTrackFreqSeq import ExtractTrackFreq

def ExtractSentProbWiki(args,file_name):
    with open(file_name,'r') as f:
        reader = csv.reader(f)
        file = [row for row in reader]
    head = file[0]
    text = file[1:]
    probs = [float(line[head.index('prob_{args.temp}')]) for line in text]
    return probs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--chain_len', type=int)
    parser.add_argument('--sent_sample', type=int)
    parser.add_argument('--num_tokens', type=int)
    parser.add_argument('--temp', type=float)
    args = parser.parse_args()
    print('running with args', args)
    
    #Extract sent probs for BERT
    path = f'BertData/{args.num_tokens}TokenSents/textfile/{args.model}/{args.batch_size}_{args.chain_len}/'
    files = [file_name.replace(path,'').replace('.csv','') for file_name in glob.glob(f'{path}*.csv')]
    
    func_args = [(args,file_name,path) for file_name in files]
    num_init_sents = 40
    num_chain_per_file = num_init_sents//len(func_args)
    extracted_chain_len = args.chain_len//args.sent_sample
    args.num_chain_per_file = num_chain_per_file
    args.extracted_chain_len = extracted_chain_len
    args.metric = 'prob'
    print(f'# of initial sentences per file: {num_chain_per_file}')
    print(f'# of sentences per chain: {extracted_chain_len}')
    
    with Pool(processes=20) as p:
        FreqCountList = p.starmap(ExtractTrackFreq,func_args)
    BertSentProb = np.array(FreqCountList).reshape((num_init_sents,len(FreqCountList[0][0]),len(FreqCountList[0][0][0])))

    #Extract sent probs for Wikipedia
    path = f'WikiData/TokenSents/{args.num_tokens}TokenSents/textfile/SentProbs/'
    files = [file_name for file_name in glob.glob(f'{path}*.csv')]

    func_args = [(args,file_name) for file_name in files]
    with Pool(processes=50) as p:
        SentProbList = p.starmap(ExtractSentProbWiki,func_args)
    WikiSentProb = np.array([prob for ProbList in SentProbList for prob in ProbList])
    print(f'# of wiki sentences: {len(WikiSentProb)}')
    WikiAve = np.array([np.mean(WikiSentProb) for i in range(extracted_chain_len)])
    WikiSem = np.array([np.std(WikiSentProb)/np.sqrt(len(WikiSentProb)) for i in range(extracted_chain_len)])
    BertAve = np.mean(BertSentProb,axis=1)
    BertSem = np.std(BertSentProb,axis=1)/np.sqrt(BertSentProb.shape[1])

    color_list = sns.color_palette('Set2')
    color_seq = sns.color_palette("coolwarm",n_colors=40)
    fig = plt.figure(figsize=(15,5),dpi=200)
    for chain_num,ave,sem in zip(np.arange(BertSentProb.shape[0]),BertAve,BertSem):
        plt.plot(ave, color=color_seq[chain_num], linewidth=1)
        plt.fill_between(np.arange(extracted_chain_len), ave-sem, ave+sem, color=color_seq[chain_num], alpha = 0.2)
    plt.plot(WikiAve, color=color_list[0], linewidth=1, label='Wikipedia')
    plt.fill_between(np.arange(extracted_chain_len), WikiAve-WikiSem, WikiAve+WikiSem, color=color_list[0], alpha = 0.5)
    plt.legend()
    fig.savefig(f'figures/SentProb_{args.model}_{args.batch_size}_{args.chain_len}_{args.sent_sample}.png')

    fig = plt.figure(figsize=(15,5),dpi=200)
    for chain_num,FreqCount in enumerate(BertSentProb):
        plt.plot(FreqCount.T, color=color_seq[chain_num], linewidth=1, alpha=0.05)
    plt.plot(WikiAve, color=color_list[0], linewidth=1, label='Wikipedia')
    plt.fill_between(np.arange(extracted_chain_len), WikiAve-WikiSem, WikiAve+WikiSem, color=color_list[0], alpha=0.8)
    plt.legend()
    fig.savefig(f'figures/SentProb_{args.model}_{args.batch_size}_{args.chain_len}_{args.sent_sample}_all.png')

    fig = plt.figure(figsize=(10,5),dpi=200)
    plt.hist(np.exp(WikiSentProb), bins=np.arange(0,1,0.001), density=True, histtype='step', label='Wikipedia')
    BertAll = BertSentProb[:,:,200:].ravel()
    print(f'# of bert sentences used in the histogram: {len(BertAll)}')
    print(f'Log likelihood for Wikipedia: min {np.min(WikiSentProb)}, max {np.max(WikiSentProb)}')
    print(f'Log likelihood for Bert: min {np.min(BertAll)}, max {np.max(BertAll)}')
    plt.hist(np.exp(BertAll), bins=np.arange(0,1,0.001), density=True, histtype='step', label='Bert')
    plt.legend(loc='upper left')
    plt.title(f'Wikipedia range: [{np.min(WikiSentProb)}, {np.max(WikiSentProb)}]\n Bert range: [{np.min(BertAll)}, {np.max(BertAll)}]')
    fig.savefig(f'figures/SentProb_{args.model}_{args.batch_size}_{args.chain_len}_{args.sent_sample}_hist.png')

    fig = plt.figure(figsize=(10,5),dpi=200)
    plt.hist(WikiSentProb, bins=np.arange(-150,1,1), density=True, histtype='step', label='Wikipedia')
    BertAll = BertSentProb[:,:,200:].ravel()
    print(f'# of bert sentences used in the histogram: {len(BertAll)}')
    print(f'Log likelihood for Wikipedia: min {np.min(WikiSentProb)}, max {np.max(WikiSentProb)}')
    print(f'Log likelihood for Bert: min {np.min(BertAll)}, max {np.max(BertAll)}')
    plt.hist(BertAll, bins=np.arange(-150,1,1), density=True, histtype='step', label='Bert')
    plt.yscale('log')
    plt.legend(loc='upper left')
    plt.title(f'Wikipedia range: [{np.min(WikiSentProb)}, {np.max(WikiSentProb)}]\n Bert range: [{np.min(BertAll)}, {np.max(BertAll)}]')
    fig.savefig(f'figures/SentProb_{args.model}_{args.batch_size}_{args.chain_len}_{args.sent_sample}_hist_log.png')
