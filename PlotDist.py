import numpy as np
import pickle
import matplotlib.pyplot as plt
import csv
import sys
import os
from collections import OrderedDict
import seaborn as sns
sns.set()

sys.path.append('..')
args = sys.argv

metric = args[1]
with open(f'10WordSents/{metric.upper()}FreqAll.pkl','rb') as f:
    wiki_all = pickle.load(f)
with open(f'10WordSents/{metric.upper()}FreqHalf.pkl','rb') as f:
    wiki_half = pickle.load(f)
with open(f'datafile/{metric.upper()}FreqAllBert.pkl','rb') as f:
    bert_all = pickle.load(f)

wiki_all_sum = np.array(list(wiki_all.values())).sum()
wiki_half_sum = [np.array(list(wiki_half_dict.values())).sum() for wiki_half_dict in wiki_half]
bert_all_sum = np.array(list(bert_all.values())).sum()

assert metric in ['pos','tag','dep'], 'Invalid metric name'
if metric == 'dep':
    max_key = np.max(list(wiki_all.keys()))
    wiki_all_plot = np.zeros(max_key+1)
    for key in wiki_all.keys():
        wiki_all_plot[key] = wiki_all[key]/wiki_all_sum
    max_key = np.max(list(bert_all.keys()))
    bert_all_plot = np.zeros(max_key+1)
    for key in bert_all.keys():
        bert_all_plot[key] = bert_all[key]/bert_all_sum

    normalized_wiki_half = {}
    for iter,wiki_half_dict in enumerate(wiki_half):
        for key in wiki_half_dict.keys():
            if key not in normalized_wiki_half:
                normalized_wiki_half[key] = []
            normalized_wiki_half[key].append(wiki_half_dict[key]/wiki_half_sum[iter])
    max_key = np.max(list(normalized_wiki_half.keys()))
    wiki_half_plot_ave = np.zeros(max_key+1)
    wiki_half_plot_sem = np.zeros(max_key+1)
    wiki_half_plot_max = np.zeros(max_key+1)
    wiki_half_plot_min = np.zeros(max_key+1)
    for key,value in normalized_wiki_half.items():
        wiki_half_plot_ave[key] = np.array(value).mean()
        wiki_half_plot_sem[key] = np.array(value).std()
        wiki_half_plot_max[key] = np.array(value).max()
        wiki_half_plot_min[key] = np.array(value).min()
    fig = plt.figure(figsize=(10,5),dpi=200)
    plt.plot(np.arange(len(wiki_all_plot))+1,wiki_all_plot,label='wiki_all')
    plt.plot(np.arange(len(bert_all_plot))+1,bert_all_plot,label='bert')
    #plt.errorbar(np.arange(max_key+1)+1,wiki_half_plot_ave,yerr=wiki_half_plot_sem,label='wiki_half')
    plt.plot(np.arange(len(wiki_all_plot))+1,wiki_half_plot_max,label='wiki_half_max')
    plt.plot(np.arange(len(wiki_all_plot))+1,wiki_half_plot_min,label='wiki_half_min')
    plt.xscale('log')
    plt.legend()
else:
    key_sets = [set(wiki_all.keys()), set(bert_all.keys())]
    data_labels = ['wiki_all','bert']
    for i,set_1 in enumerate(key_sets):
        for j,set_2 in enumerate(key_sets):
            print(f'{data_labels[i]} - {data_labels[j]}: {set_1-set_2}')

    wiki_all_ordered = OrderedDict(wiki_all)
    wiki_all_ordered = OrderedDict(sorted(wiki_all_ordered.items(), key=lambda x: x[1], reverse=True))
    labels = list(wiki_all_ordered.keys())
    if metric == 'pos':
        labels.remove('SPACE')
    elif metric == 'tag':
        labels.remove('_SP')

    plot_data = np.empty((len(labels),2))
    for i,label in enumerate(labels):
        plot_data[i,0] = wiki_all[label]/wiki_all_sum
        plot_data[i,1] = bert_all[label]/bert_all_sum

    half_plot_data = np.empty((len(wiki_half),len(labels)))
    for iter, wiki_half_dict in enumerate(wiki_half):
        for i,label in enumerate(labels):
            half_plot_data[iter][i] = wiki_half_dict[label]/wiki_half_sum[iter]

    fig = plt.figure(figsize=(10,5),dpi=200)
    for i,label in enumerate(data_labels):
        plt.plot(plot_data[:,i],label=label)
    #plt.errorbar(np.arange(len(labels)), np.mean(half_plot_data,axis=0), yerr=np.std(half_plot_data,axis=0), label='wiki_half')
    plt.plot(np.arange(len(labels)), np.max(half_plot_data,axis=0), label='wiki_half_max')
    plt.plot(np.arange(len(labels)), np.min(half_plot_data,axis=0), label='wiki_half_min')
    plt.xticks(np.arange(len(labels)),labels,rotation=45,fontsize=7)
plt.legend()
fig.savefig(f'figures/{metric.upper()}Half.png')

