import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
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
with open(f'datafile/cond_prob_array_{sampling_method}_{sentence_id}_{temp}.pkl','rb') as f:
    cond_prob = pickle.load(f)
cond_prob = np.array([np.average(row,axis=-1) for row in cond_prob])
print(edit_rate.shape)
batch_num = edit_rate.shape[0]
batch_size = edit_rate.shape[1]
chain_len = edit_rate.shape[2]
prob_sample = int(chain_len/prob.shape[-1])
fig = plt.figure()
color_list = sns.color_palette("Paired")
for i in range(batch_num):
    for j in range(batch_size):
        plt.plot(edit_rate[i][j],color=color_list[i//(batch_num//2)*2],alpha=0.1)
plt.plot(np.average(edit_rate[:batch_num//2].reshape((int(batch_num*batch_size/2),chain_len)),axis=0),color=color_list[1],linewidth=3,label='High Probability')
plt.plot(np.average(edit_rate[batch_num//2:].reshape((int(batch_num*batch_size/2),chain_len)),axis=0),color=color_list[3],linewidth=3,label='Low Probability')
plt.legend()
plt.xlabel("Time step")
plt.ylabel("Edit rate")
plt.show()
fig = plt.figure()
for i in range(batch_num):
    for j in range(batch_size):
        plt.plot([prob_sample*i for i in range(int(chain_len/prob_sample))], prob[i][j],color=color_list[i//(batch_num//2)*2],alpha=0.2)
plt.plot([prob_sample*i for i in range(int(chain_len/prob_sample))], np.average(prob[:batch_num//2].reshape((int(batch_num*batch_size/2),prob.shape[-1])),axis=0),color=color_list[1],linewidth=3,label='High Probability')
plt.plot([prob_sample*i for i in range(int(chain_len/prob_sample))], np.average(prob[batch_num//2:].reshape((int(batch_num*batch_size/2),prob.shape[-1])),axis=0),color=color_list[3],linewidth=3,label='Low Probability')
plt.legend()
plt.xlabel("Time step")
plt.ylabel("Sentence probability")
plt.show()

fig = plt.figure()
for i in range(batch_num):
    for j in range(batch_size):
        plt.plot(cond_prob[i][j],color=color_list[i//(batch_num//2)*2],alpha=0.2)
plt.plot(np.average(cond_prob[:batch_num//2].reshape((int(batch_num*batch_size/2),cond_prob.shape[-1])),axis=0),color=color_list[1],linewidth=3,label='High Probability')
plt.plot(np.average(cond_prob[batch_num//2:].reshape((int(batch_num*batch_size/2),cond_prob.shape[-1])),axis=0),color=color_list[3],linewidth=3,label='Low Probability')
plt.legend()
plt.xlabel("Time step")
plt.ylabel("Conditional probability")
plt.show()
