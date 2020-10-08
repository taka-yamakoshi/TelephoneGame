import numpy as np
import pickle
import matplotlib.pyplot as plt
import csv
import sys
import os
from collections import OrderedDict
from similaritymeasures import area_between_two_curves
import seaborn as sns
sns.set()

sys.path.append('..')
args = sys.argv

def CalcFreqFromDict(file_name):
    with open(file_name,'rb') as f:
        loaded_dict = pickle.load(f)

    loaded_dict = OrderedDict(loaded_dict)
    loaded_dict = OrderedDict(sorted(loaded_dict.items(), key=lambda x: x[1], reverse=True))


    extracted_words = list(loaded_dict.keys())
    extracted_freq = np.array(list(loaded_dict.values()))
    total_freq = np.sum(extracted_freq)
    extracted_freq_density = extracted_freq/total_freq
    return loaded_dict, extracted_words, extracted_freq_density,total_freq


dict_wiki_all, words_wiki_all, freq_wiki_all, total_freq_wiki_all = CalcFreqFromDict('10WordSents/VOCABFreqAll.pkl')
_, words_wiki_A, freq_wiki_A, _ = CalcFreqFromDict('10WordSents/VOCABFreqHalfA.pkl')
_, words_wiki_B, freq_wiki_B, _ = CalcFreqFromDict('10WordSents/VOCABFreqHalfB.pkl')
dict_bert, words_bert, freq_bert, total_freq_bert = CalcFreqFromDict('datafile/VOCABFreqAllBert.pkl')

'''
fig = plt.figure(figsize=[8,6],dpi=150)
plt.plot(np.arange(len(freq_wiki_all))+1,freq_wiki_all,label='Wikipedia')
plt.plot(np.arange(len(freq_bert))+1,freq_bert,label='BERT')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Frequency Rank')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
fig.savefig('figures/ZipfAll.png')
plt.close()
'''

'''
fig = plt.figure(figsize=[8,6],dpi=150)
plt.plot(np.arange(len(freq_wiki_A))+1,freq_wiki_A,label='Wikipedia_A')
plt.plot(np.arange(len(freq_wiki_B))+1,freq_wiki_B,label='Wikipedia_B')
plt.plot(np.arange(len(freq_bert))+1,freq_bert,label='BERT')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Frequency Rank')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
fig.savefig('figures/ZipfHalf.png')
plt.close()
'''

wiki_all_data = np.array([np.log(np.arange(len(freq_wiki_all))+1),np.log(freq_wiki_all)])
wiki_A_data = np.array([np.log(np.arange(len(freq_wiki_A))+1),np.log(freq_wiki_A)])
wiki_B_data = np.array([np.log(np.arange(len(freq_wiki_B))+1),np.log(freq_wiki_B)])
bert_data = np.array([np.log(np.arange(len(freq_bert))+1),np.log(freq_bert)])
print(f'Area between WikiAll and BERT: {area_between_two_curves(wiki_all_data,bert_data)}')
print(f'Area between WikiA and BERT: {area_between_two_curves(wiki_A_data,bert_data)}')
print(f'Area between WikiB and BERT: {area_between_two_curves(wiki_B_data,bert_data)}')
print(f'Area between WikiA and WikiB: {area_between_two_curves(wiki_A_data,wiki_B_data):.15f}')


print(f'Number of words in WikiData: {len(words_wiki_all)}')
print(f'Number of words extracted from BERT: {len(words_bert)}')

WikiBERTDiff = set(words_wiki_all) - set(words_bert)
BERTWikiDiff = set(words_bert) - set(words_wiki_all)
print(f'# of words included in Wiki but not in BERT: {len(WikiBERTDiff)}')
print(f'# of words included in BERT but not in Wiki: {len(BERTWikiDiff)}')

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
