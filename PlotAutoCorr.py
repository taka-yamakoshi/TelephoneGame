import numpy as np
import pickle
import csv
import sys
import time
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

def CalcCorr(prob_array,shift):
    return np.array([[np.corrcoef(line[:-shift],line[shift:])[0,1]\
                      for line in row]\
                     for row in prob_array])

if __name__ == '__main__':
    sent_sample = 5
    num_init_sent = 10
    num_tokens = 10
    sampling_method = 'mix_random'
    temp = 1
    chain_len = 51000
    batch_size = 2
    num_masks = 2
    mc_sampling_size = 1000
    
    mixing_rates = np.array([0.1,0.2,0.5,0.8])
    lags = np.array([1,2,4,10,20,40,100,200,400,1000])
    fig = plt.figure(dpi=150,figsize=(8,6))
    auto_corr_mean_array = np.empty((len(lags),len(mixing_rates)))
    auto_corr_upper_array = np.empty((len(lags),len(mixing_rates)))
    auto_corr_lower_array = np.empty((len(lags),len(mixing_rates)))
    for rate_id,rate in enumerate(mixing_rates):
        file_name = f'BertData/{num_tokens}TokenSents/textfile/bert-base-uncased/{batch_size}_{chain_len}/'\
        +f'bert_{sampling_method}_{rate}_{mc_sampling_size}_{num_masks}_input_{temp}.csv'

        with open(file_name,'r') as f:
            reader = csv.reader(f)
            csv_file = [row for row in reader]
            head = csv_file[0]
            text = csv_file[1:]
        
        prob_array = np.array([[[float(row[head.index('prob')]) for row in text\
                                 if int(row[head.index('chain_num')])==chain_num\
                                 and int(row[head.index('sentence_num')])==sentence_num\
                                 and int(row[head.index('iter_num')]) in np.arange(1000,chain_len,sent_sample)]\
                                for chain_num in range(batch_size)] for sentence_num in range(num_init_sent)])
        iter_num_array = np.array([[[int(row[head.index('iter_num')]) for row in text\
                                     if int(row[head.index('chain_num')])==chain_num\
                                     and int(row[head.index('sentence_num')])==sentence_num\
                                     and int(row[head.index('iter_num')]) in np.arange(1000,chain_len,sent_sample)]\
                                    for chain_num in range(batch_size)] for sentence_num in range(num_init_sent)])
        print(prob_array.shape)
        for sentence_num in range(num_init_sent):
            for chain_num in range(batch_size):
                assert all(iter_num_array[sentence_num,chain_num]==np.arange(1000,chain_len,sent_sample))
                
        auto_corr_mean = []
        auto_corr_upper = []
        auto_corr_lower = []
        for lag in lags:
            sampled_prob_array = np.array([[line[np.arange(0,prob_array.shape[-1],lag)] for line in row] for row in prob_array])
            print(sampled_prob_array.shape)
            corr = CalcCorr(sampled_prob_array,1).ravel()
            corr[np.isnan(corr)] = 1
            auto_corr_mean.append(corr.mean())
            auto_corr_upper.append(corr.mean()+corr.std()/np.sqrt(corr.size))
            auto_corr_lower.append(corr.mean()-corr.std()/np.sqrt(corr.size))
        auto_corr_mean_array[:,rate_id] = auto_corr_mean
        auto_corr_upper_array[:,rate_id] = auto_corr_upper
        auto_corr_lower_array[:,rate_id] = auto_corr_lower
    for mean,upper,lower,lag in zip(auto_corr_mean_array,auto_corr_upper_array,auto_corr_lower_array,lags):
        plt.plot(mixing_rates*100,mean,label=lag*sent_sample)
        plt.fill_between(mixing_rates*100,lower,upper,alpha=0.5)
    plt.xlabel('% MH')
    plt.ylabel('average auto correlation')
    fig.tight_layout(rect=[0,0,0.8,1])
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    fig.savefig('figures/auto_corr_mix.png')