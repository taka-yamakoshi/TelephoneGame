# export WIKI_PATH='YOUR PATH TO WIKICORPUS'
# export BOOK_PATH='YOUR PATH TO BOOKCORPUS'
import numpy as np
import pickle
import csv
from multiprocessing import Pool
import os
import argparse
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, required=True, choices=['wiki','book'])
    args = parser.parse_args()

    color_list = sns.color_palette('Set2')

    if args.corpus=='wiki':
        text_path = f'{os.environ.get("WIKI_PATH")}/TokenSents/'
    elif args.corpus=='book':
        text_path = f'{os.environ.get("BOOK_PATH")}/TokenSents/'

    folder_path_list = glob.glob(f'{text_path}*TokenSents')
    sent_num_dict = {}
    for folder_path in tqdm(folder_path_list,ncols=100):
        num_tokens = int(folder_path.replace(text_path,'').replace('TokenSents',''))
        files = glob.glob(f'{folder_path}/textfile/*.txt')
        num_sents = 0
        for file in files:
            with open(file,'r') as f:
                text = f.read().split('\n')[:-1]
            num_sents += len(text)
        sent_num_dict[num_tokens] = num_sents
    fig = plt.figure()
    plot_data = np.array([[key,value] for key,value in sent_num_dict.items()])
    if args.corpus=='wiki':
        sent_num_list = np.array([list(plot_data[:,0]).index(i) for i in [12,21,37]])
    elif args.corpus=='book':
        sent_num_list = np.array([list(plot_data[:,0]).index(i) for i in [11]])
    plt.scatter(plot_data[sent_num_list,0],plot_data[sent_num_list,1]/np.sum(plot_data[:,1]),color=color_list[1])
    plt.scatter(plot_data[:,0],plot_data[:,1]/np.sum(plot_data[:,1]),marker='.',color=color_list[0])
    plt.xscale('log')
    if args.corpus=='wiki':
        plt.title(f'Number of BERT tokens for {args.corpus.capitalize()} corpus\n in range [{np.min(plot_data[:,0])},{np.max(plot_data[:,0])}] with mode at {plot_data[np.argmax(plot_data[:,1]),0]}\n Orage dots at [12,21,37]')
    elif args.corpus=='book':
        plt.title(f'Number of BERT tokens for {args.corpus.capitalize()} corpus\n in range [{np.min(plot_data[:,0])},{np.max(plot_data[:,0])}] with mode at {plot_data[np.argmax(plot_data[:,1]),0]}\n Orage dot at 11')
    fig.savefig(f'figures/{args.corpus.capitalize()}SentNums.png')
