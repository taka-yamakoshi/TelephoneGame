import numpy as np
import pickle
import matplotlib.pyplot as plt
import csv
import sys
import os
from collections import OrderedDict, Counter
from sklearn.utils import resample
import seaborn as sns
sns.set()

sys.path.append('..')
args = sys.argv

metric = args[1]
with open(f'10WordSents/{metric.upper()}FreqAll.pkl','rb') as f:
    wiki_all = pickle.load(f)
with open(f'datafile/{metric.upper()}FreqAllBert.pkl','rb') as f:
    bert_all = pickle.load(f)

def Bootstrap(init_dict,iter_num,metric,specified_order):
    init_list = []
    for word,value in init_dict.items():
        init_list.extend([word for i in range(value)])
    return np.array([CreatePlotListFromDict(Counter(resample(init_list)),metric,specified_order) for i in range(iter_num)])

def CreatePlotListFromDict(init_dict,metric,specified_order):
    value_sum = np.array(list(init_dict.values())).sum()
    if metric == 'dep':
        max_key = np.max(list(init_dict.keys()))
        new_array = np.zeros(max_key+1)
        for key,value in init_dict.items():
            new_array[key] = value/value_sum
    else:
        new_array = []
        for word in specified_order:
            new_array.append(init_dict[word]/value_sum)
        new_array = np.array(new_array)
    return new_array

assert metric in ['pos','tag','dep'], 'Invalid metric name'
color_list = sns.color_palette('Set2')
iter_num = 20
if metric == 'dep':
    wiki_all_plot = CreatePlotListFromDict(wiki_all,'dep',None)
    bert_all_plot = CreatePlotListFromDict(bert_all,'dep',None)
    wiki_bootstrap = Bootstrap(wiki_all,iter_num,'dep',None)
    bert_bootstrap = Bootstrap(bert_all,iter_num,'dep',None)
    fig = plt.figure(figsize=(10,5),dpi=200)
    plt.plot(np.arange(len(wiki_all_plot))+1,wiki_all_plot,label='wiki',color=color_list[0])
    plt.plot(np.arange(len(bert_all_plot))+1,bert_all_plot,label='bert',color=color_list[1])
    for wiki_bootstrap_iter, bert_bootstrap_iter in zip(wiki_bootstrap,bert_bootstrap):
        plt.plot(np.arange(len(wiki_bootstrap_iter))+1,wiki_bootstrap_iter,alpha=0.5,color=color_list[2])
        plt.plot(np.arange(len(bert_bootstrap_iter))+1,bert_bootstrap_iter,alpha=0.5,color=color_list[3])
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
    wiki_all_plot = CreatePlotListFromDict(wiki_all,metric,labels)
    bert_all_plot = CreatePlotListFromDict(bert_all,metric,labels)
    wiki_bootstrap = Bootstrap(wiki_all,iter_num,metric,labels)
    bert_bootstrap = Bootstrap(bert_all,iter_num,metric,labels)
    fig = plt.figure(figsize=(10,5),dpi=200)
    plt.plot(np.arange(len(labels)),wiki_all_plot,color=color_list[0],label='wiki')
    plt.plot(np.arange(len(labels)),bert_all_plot,color=color_list[1],label='bert')
    for wiki_bootstrap_iter, bert_bootstrap_iter in zip(wiki_bootstrap,bert_bootstrap):
        plt.plot(np.arange(len(labels)),wiki_bootstrap_iter,alpha=0.5,color=color_list[2])
        plt.plot(np.arange(len(labels)),bert_bootstrap_iter,alpha=0.5,color=color_list[3])
    #plt.errorbar(np.arange(len(labels)), np.mean(half_plot_data,axis=0), yerr=np.std(half_plot_data,axis=0), label='wiki_half')
    #plt.plot(np.arange(len(labels)), np.max(half_plot_data,axis=0), label='wiki_half_max')
    #plt.plot(np.arange(len(labels)), np.min(half_plot_data,axis=0), label='wiki_half_min')
    plt.xticks(np.arange(len(labels)),labels,rotation=45,fontsize=7)
plt.legend()
fig.savefig(f'figures/{metric.upper()}Bootstrap.png')

