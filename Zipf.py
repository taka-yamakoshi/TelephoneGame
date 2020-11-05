import numpy as np
import pickle
import matplotlib.pyplot as plt
import csv
import sys
import os
from collections import OrderedDict
#from similaritymeasures import area_between_two_curves
import seaborn as sns
sns.set()

sys.path.append('..')
args = sys.argv

def CalcFreqFromDict(loaded_dict):
    loaded_dict = OrderedDict(loaded_dict)
    loaded_dict = OrderedDict(sorted(loaded_dict.items(), key=lambda x: x[1], reverse=True))


    extracted_words = list(loaded_dict.keys())
    extracted_freq = np.array(list(loaded_dict.values()))
    total_freq = np.sum(extracted_freq)
    extracted_freq_density = extracted_freq/total_freq
    return [loaded_dict, extracted_words, extracted_freq_density,total_freq]


with open('../WikiData/10WordSents/VOCABFreqAll.pkl','rb') as f:
    wiki_all = pickle.load(f)
with open('datafile/VOCABFreqAllBert.pkl','rb') as f:
    bert_all = pickle.load(f)
[dict_wiki_all, words_wiki_all, freq_wiki_all, total_freq_wiki_all] = CalcFreqFromDict(wiki_all)
[dict_bert, words_bert, freq_bert, total_freq_bert] = CalcFreqFromDict(bert_all)

with open('../WikiData/10WordSents/VOCABFreqHalf.pkl','rb') as f:
    wiki_half = pickle.load(f)
normalized_freq_half = {}
for wiki_half_dict in wiki_half:
    freq_wiki_half = CalcFreqFromDict(wiki_half_dict)[2]
    for i, element in enumerate(freq_wiki_half):
        if i not in normalized_freq_half:
            normalized_freq_half[i] = []
        normalized_freq_half[i].append(element)
max_key = len(list(normalized_freq_half.keys()))
plot_wiki_half_ave = np.zeros(max_key+1)
plot_wiki_half_sem = np.zeros(max_key+1)
plot_wiki_half_max = np.zeros(max_key+1)
plot_wiki_half_min = np.zeros(max_key+1)
for key,value in normalized_freq_half.items():
    plot_wiki_half_ave[key] = np.array(value).mean()
    plot_wiki_half_sem[key] = np.array(value).std()
    plot_wiki_half_max[key] = np.array(value).max()
    plot_wiki_half_min[key] = np.array(value).min()

color_list = sns.color_palette('Set2')
fig = plt.figure(figsize=[8,6],dpi=150)
plt.plot(np.arange(len(freq_wiki_all))+1, freq_wiki_all, label='Wikipedia', color=color_list[0])
plt.plot(np.arange(len(freq_bert))+1, freq_bert, label='BERT', color=color_list[1])
plt.plot(np.arange(max_key+1)+1, plot_wiki_half_max, label='Wikipedia Half Max', color=color_list[0], alpha=0.5)
plt.plot(np.arange(max_key+1)+1, plot_wiki_half_min, label='Wikipedia Half Min', color=color_list[0], alpha=0.5)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Frequency Rank')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
fig.savefig('figures/ZipfAll.png')
plt.close()

'''
wiki_all_data = np.array([np.log(np.arange(len(freq_wiki_all))+1),np.log(freq_wiki_all)])
bert_data = np.array([np.log(np.arange(len(freq_bert))+1),np.log(freq_bert)])
print(f'Area between WikiAll and BERT: {area_between_two_curves(wiki_all_data,bert_data)}')
'''

print(f'Number of words in WikiData: {len(words_wiki_all)}')
print(f'Number of words extracted from BERT: {len(words_bert)}')

WikiBERTDiff = set(words_wiki_all) - set(words_bert)
BERTWikiDiff = set(words_bert) - set(words_wiki_all)
print(f'# of words included in Wiki but not in BERT: {len(WikiBERTDiff)}')
print(f'# of words included in BERT but not in Wiki: {len(BERTWikiDiff)}')
with open('BERT-Wiki.txt','w') as f:
    for element in BERTWikiDiff:
        f.write(str(element))
        f.writelines('\n')
with open('Wiki-BERT.txt','w') as f:
    for element in WikiBERTDiff:
        f.write(str(element))
        f.writelines('\n')

print('Calculating frequency')
FreqWikiBERTDiff = np.log10(np.array([dict_wiki_all[word] for word in WikiBERTDiff])/total_freq_wiki_all)
FreqBERTWikiDiff = np.log10(np.array([dict_bert[word] for word in BERTWikiDiff])/total_freq_bert)

print('Plotting')
fig = plt.figure(figsize=[8,6],dpi=150)
plt.hist(FreqWikiBERTDiff,bins=np.arange(-8,1,1),label='Wiki-BERT',histtype='step',log=True)
plt.hist(FreqBERTWikiDiff,bins=np.arange(-8,1,1),label='BERT-Wiki',histtype='step',log=True)
plt.xlabel('Frequency in corpus')
plt.ylabel('Frequency in difference set')
plt.legend(loc='upper right')
fig.savefig('figures/ZipfDiff.png')
plt.close()

