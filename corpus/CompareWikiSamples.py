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
from CountFreq import ExtractFreqNew, TokenizerSetUpNew
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

def ExtractFeatures(folder_path,tokenizer,args):
    with open(f'{folder_path}','r') as f:
        reader = csv.reader(f)
        file = [row for row in reader]
    head = file[0]
    text = file[1:]
    probs = [float(row[head.index('prob_1')]) for row in text]
    sentences = [row[head.index('sentence')] for row in text]
    args.metric = 'pos'
    pos_freq,_,sent_num_pos = ExtractFreqNew(sentences,tokenizer,sent_tokenize,nlp,args,verbose=True)
    
    doc = nlp.pipe(sentences)
    args.metric = 'tag'
    tag_freq,_,sent_num_tag = ExtractFreqNew(sentences,tokenizer,sent_tokenize,nlp,args,verbose=True)
    
    doc = nlp.pipe(sentences)
    args.metric = 'dep'
    dep_freq,_,sent_num_dep = ExtractFreqNew(sentences,tokenizer,sent_tokenize,nlp,args,verbose=True)
    print(len(sentences),sent_num_pos,sent_num_tag,sent_num_dep)
    return (probs,pos_freq,tag_freq,dep_freq)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, choices=['wiki','book'])
    parser.add_argument('--num_tokens',type=int, required = True,
                        help='number of tokens including special tokens')
    parser.add_argument('--model',type=str, default='bert-base-uncased')
    args = parser.parse_args()
    if args.corpus=='wiki':
        text_path = f'wikicorpus/TokenSents/{args.num_tokens}TokenSents/textfile/'
        data_path = f'wikicorpus/TokenSents/{args.num_tokens}TokenSents/datafile/'
    elif args.corpus=='book':
        text_path = f'bookcorpus/TokenSents/{args.num_tokens}TokenSents/textfile/'
        data_path = f'bookcorpus/TokenSents/{args.num_tokens}TokenSents/datafile/'
    
    # Extract sent_prob from all Wikipedia sentences with num_tokens = args.num_tokens
    with open(f'{text_path}SentProbs/sentences.csv','r') as f:
        reader = csv.reader(f)
        file = [row for row in reader]
        head = file[0]
        text = file[1:]
        prob_all = np.array([float(row[head.index('prob_1')]) for row in text])
    print(prob_all.shape)
    print(prob_all[:10])
    


    '''
    # Load feature freq of all Wikipedia sentences with num_tokens = args.num_tokens
    with open(f'{data_path}POSFreqAll.pkl','rb') as f:
        pos_all = pickle.load(f)
    with open(f'{data_path}TAGFreqAll.pkl','rb') as f:
        tag_all = pickle.load(f)
    with open(f'{data_path}DEPFreqAll.pkl','rb') as f:
        dep_all = pickle.load(f)
    '''

    # Extract sent_prob and feature freq from sampled Wikipedia sentences
    nlp = TokenizerSetUpNew()
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model)
    from nltk.tokenize import sent_tokenize
    
    folder_path_list = glob.glob(f'{text_path}SampledSents/*.csv')
    arg = [(folder_path,tokenizer,args) for folder_path in folder_path_list]
    with Pool(processes=50) as p:
        freq_dict_list = p.starmap(ExtractFeatures,arg)
    
    prob_sample = [line[0] for line in freq_dict_list]
    '''
    pos_sample = [line[1] for line in freq_dict_list]
    tag_sample = [line[2] for line in freq_dict_list]
    dep_sample = [line[3] for line in freq_dict_list]
    '''

    
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

    
    # Compare POS
    fig = plt.figure(dpi=150,figsize=(10,10))
    pos_list = list(pos_all.keys())
    print(f'pos_list: {pos_list}')
    pos_plot_data_all = np.array([pos_all[pos] for pos in pos_list])
    pos_plot_data_all = pos_plot_data_all/pos_plot_data_all.sum()
    pos_plot_data_sample = []
    for sample in pos_sample:
        plot_data = np.array([sample[pos] if pos in sample else 0 for pos in pos_list])
        plot_data = plot_data/plot_data.sum()
        plt.plot(plot_data,color=color_list[0],alpha=0.2)
        pos_plot_data_sample.append(plot_data)
    pos_plot_data_sample = np.array(pos_plot_data_sample)
    plt.plot(pos_plot_data_all,color=color_list[1],label='all')
    plt.plot(pos_plot_data_sample[diff.argmin()],color=color_list[2],label='closest')
    plt.xticks(np.arange(len(pos_list)),pos_list,rotation=45)
    plt.title(f'POS distribution for {args.num_tokens} token sentences\nChosen sample: {folder_path_list[diff.argmin()].replace(f"{text_path}SampledSents/","")}')
    plt.xlabel('POS labels')
    plt.legend()
    fig.savefig(f'figures/sample_comparison_{args.corpus}_{args.num_tokens}_pos.png')
    
    # Compare TAG
    fig = plt.figure(dpi=150,figsize=(10,10))
    tag_list = list(tag_all.keys())
    print(f'tag_list: {tag_list}')
    tag_plot_data_all = np.array([tag_all[tag] for tag in tag_list])
    tag_plot_data_all = tag_plot_data_all/tag_plot_data_all.sum()
    tag_plot_data_sample = []
    for sample in tag_sample:
        plot_data = np.array([sample[tag] if tag in sample else 0 for tag in tag_list])
        plot_data = plot_data/plot_data.sum()
        plt.plot(plot_data,color=color_list[0],alpha=0.2)
        tag_plot_data_sample.append(plot_data)
    tag_plot_data_sample = np.array(tag_plot_data_sample)
    plt.plot(tag_plot_data_all,color=color_list[1],label='all')
    plt.plot(tag_plot_data_sample[diff.argmin()],color=color_list[2],label='closest')
    plt.xticks(np.arange(len(tag_list)),tag_list,rotation=45)
    plt.title(f'TAG distribution for {args.num_tokens} token sentences\nChosen sample: {folder_path_list[diff.argmin()].replace(f"{text_path}SampledSents/","")}')
    plt.xlabel('TAG labels')
    plt.legend()
    fig.savefig(f'figures/sample_comparison_{args.corpus}_{args.num_tokens}_tag.png')
    
    # Compare DEP
    fig = plt.figure(dpi=150,figsize=(10,10))
    for sample in dep_sample:
        plt.scatter(list(sample.keys()),
                    np.array(list(sample.values()))/np.array(list(sample.values())).sum(),
                    color=color_list[0],alpha=0.2)
    plt.scatter(list(dep_all.keys()),
                np.array(list(dep_all.values()))/np.array(list(dep_all.values())).sum(),
                color=color_list[1],label='all')
    plt.scatter(list(dep_sample[diff.argmin()].keys()),
                np.array(list(dep_sample[diff.argmin()].values()))/np.array(list(dep_sample[diff.argmin()].values())).sum(),
                color=color_list[2],label='closest')
    plt.title(f'DEP distribution for {args.num_tokens} token sentences\nChosen sample: {folder_path_list[diff.argmin()].replace(f"{text_path}SampledSents/","")}')
    plt.xlabel('dependency distance')
    plt.legend()
    fig.savefig(f'figures/sample_comparison_{args.corpus}_{args.num_tokens}_dep.png')

    '''
    # Perform statistical test using area_between_curves?
    from similaritymeasures import area_between_two_curves
    assert len(np.arange(math.floor(np.min(prob_all)),0))-1==len(binned_prob_all)
    area_sent_prob = np.array([area_between_two_curves(np.array([np.arange(math.floor(np.min(prob_all)),0)[:-1],binned_prob]).T,
                                                       np.array([np.arange(math.floor(np.min(prob_all)),0)[:-1],binned_prob_all]).T
                                                      ) for binned_prob in binned_prob_sample])
    print(area_sent_prob)
    '''
    