import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
args = sys.argv

sampling_method = args[1]
sentence_id = args[2]
temp = args[3]
with open(f'datafile/edit_rate_array_{sampling_method}_{sentence_id}_{temp}.pkl','rb') as f:
    edit_rate = pickle.load(f)
with open(f'datafile/prob_array_{sampling_method}_{sentence_id}_{temp}.pkl','rb') as f:
    prob = pickle.load(f)
print(edit_rate.shape)
batch_num = edit_rate.shape[0]
batch_size = edit_rate.shape[1]
chain_len = edit_rate.shape[2]
prob_sample = int(chain_len/prob.shape[-1])
fig = plt.figure()
color_list = ['black','red','blue','green','purple']
for i in range(batch_num):
    for j in range(batch_size):
        plt.plot(edit_rate[i][j],color=color_list[i],alpha=0.1)
plt.plot(np.average(edit_rate.reshape((batch_num*batch_size,chain_len)),axis=0),color='black',linewidth=3)
plt.xlabel("Time step")
plt.ylabel("Edit rate")
plt.show()
fig = plt.figure()
for i in range(batch_num):
    for j in range(batch_size):
        plt.plot([prob_sample*i for i in range(int(chain_len/prob_sample))], prob[i][j],color=color_list[i],alpha=0.2)
plt.plot([prob_sample*i for i in range(int(chain_len/prob_sample))], np.average(prob.reshape((batch_num*batch_size,prob.shape[-1])),axis=0),color='black',linewidth=3)
plt.xlabel("Time step")
plt.ylabel("Sentence probability")
plt.show()

