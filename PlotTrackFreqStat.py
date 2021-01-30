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
import collections
from sklearn.utils import resample

def ExtractTrackFreqStat(args,file_name,path):
    with open(f'{path}{file_name}.csv','r') as f:
        reader = csv.reader(f)
        file = [row for row in reader]
    head = file[0]
    text = file[1:]
    if args.metric in ['bert_tokens','spacy_tokens','spacy_sents']:
        FreqList = [int(row[head.index(args.metric)]) for row in text]
        return FreqList

def BootstrapFreqList(i,FreqList):
    FreqDict = collections.Counter(resample(FreqList))
    return FreqDict

def CreateArrayFromDict(FreqDict):
    key_list = []
    value_list = []
    for key,value in FreqDict.items():
        key_list.append(key)
        value_list.append(value)
    return np.array([key_list,value_list])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type = str)
    parser.add_argument('--model', type = str)
    parser.add_argument('--metric', type = str)
    parser.add_argument('--batch_size', type = int)
    parser.add_argument('--chain_len', type = int)
    parser.add_argument('--sent_sample', type = int)
    parser.add_argument('--temp', type = float)
    parser.add_argument('--iter_num', type = int, default = 20)
    parser.add_argument('--num_tokens', type = int, default = 13)
    args = parser.parse_args()
    print('running with args', args)
    if args.corpus == 'bert':
        path = f'BertData/{args.num_tokens}TokenSents/textfile/{args.model}/{args.batch_size}_{args.chain_len}/TrackFreq/'
    elif args.corpus == 'wiki':
        path = f'WikiData/TokenSents/{args.num_tokens}TokenSents/textfile/TrackFreq/'
    files = [file_name.replace(f'{path}','').replace('.csv','') for file_name in glob.glob(f'{path}*_{args.temp}.csv')]
    
    func_args = [(args,file_name,path) for file_name in files]
    with Pool(processes=100) as p:
        FreqListList = p.starmap(ExtractTrackFreqStat,func_args)
    FreqListAll = []
    for FreqList in FreqListList:
        FreqListAll.extend(FreqList)
    PlotData = CreateArrayFromDict(collections.Counter(FreqListAll))
    func_args = [(i,FreqListAll) for i in range(args.iter_num)]
    with Pool(processes=100) as p:
        BootstrappedList = p.starmap(BootstrapFreqList,func_args)
    BootstrappedPlotData = [CreateArrayFromDict(FreqDict) for FreqDict in BootstrappedList]
    print(PlotData)

    color_list = sns.color_palette('Set2')
    fig = plt.figure(figsize=(10,5),dpi=200)
    for IterPlotData in BootstrappedPlotData:
        plt.scatter(IterPlotData[0],IterPlotData[1]/np.sum(IterPlotData[1]),alpha=0.05,color=color_list[1],marker='.')
    plt.scatter(PlotData[0],PlotData[1]/np.sum(PlotData[1]),color=color_list[0],marker='.')
    plt.xscale('log')
    fig.savefig(f'figures/TrackFreqStat_{args.corpus}_{args.metric}.png')
