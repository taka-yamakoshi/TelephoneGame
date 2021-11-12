import numpy as np
import pickle
import csv
from multiprocessing import Pool
import os
import argparse
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == '__main__':
    text_path = 'WikiData/TokenSents/'
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
    plt.scatter(plot_data[:,0],plot_data[:,1]/np.sum(plot_data[:,1]),marker='.')
    plt.xscale('log')
    plt.title(f'Number of BERT tokens for Wikipedia sentences\n in range [{np.min(plot_data[:,0])},{np.max(plot_data[:,0])}] with mode at {plot_data[np.argmax(plot_data[:,1]),0]}')
    fig.savefig('WikiSentNums.png')
