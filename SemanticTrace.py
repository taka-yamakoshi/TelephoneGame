import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import sys
sys.path.append('..')
args = sys.argv

sent_id = 0
temp = 0.2

with open(f'datafile/tsne_{sent_id}_{temp}.pkl','rb') as f:
    data = pickle.load(f)
print(data.shape)
(batch_num,gen_num,batch_size,dim) = data.shape
color_list = sns.color_palette("Set2")
for i in range(batch_num):
    for j in range(batch_size):
        sns.lineplot(x=data[i][:,j,0], y=data[i][:,j,1], color=color_list[i], alpha=0.2)
        sns.scatterplot(x=data[i][:1,j,0],y=data[i][:1,j,1],color='black')
        if j==0:
            sns.scatterplot(x=data[i][-1:,j,0],y=data[i][-1:,j,1],color=color_list[i],label=f'batch: {i}')
        else:
            sns.scatterplot(x=data[i][-1:,j,0],y=data[i][-1:,j,1],color=color_list[i])
plt.legend()
plt.show()
