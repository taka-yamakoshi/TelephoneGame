import torch
import numpy as np
import pickle
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import torch.nn.functional as F
import csv
import time
import sys
import os
sys.path.append('..')
args = sys.argv

def UniformGibbsSample(sentences,writer,batch_id,iter_num,decoded_init_sent):
    seq_len = sentences.shape[1]
    rand_list = np.random.permutation(seq_len-2)+1
    with torch.no_grad():
        prob = 0
        edit_num = np.zeros(sentences.shape[0])
        cond_prob = np.zeros((sentences.shape[0],len(rand_list)))
        if iter_num%prob_sample==0:
            prob = CalcSentProbBatch(sentences)
        if iter_num%sent_sample==0:
            index =chain_len*batch_id+iter_num
            writer.writerow([str(index),decoded_init_sent,str(batch_id),str(iter_num)]+[tokenizer.decode(sentence) for sentence in sentences])
        for pos_id,pos in enumerate(rand_list):
            masked_sentences = sentences.clone()
            masked_sentences[:,pos] = mask_id
            outputs = model(masked_sentences)
            probs = F.softmax(outputs[0][:,pos]/temp,dim=-1)
            chosen_words = torch.tensor([np.random.choice(len(prob),p=prob.cpu().numpy()) for prob in probs])
            if iter_num%edit_sample==0:
                for i,word in enumerate(chosen_words):
                    if word != sentences[i][pos]:
                        edit_num[i] += 1
                    cond_prob[i][pos_id] = torch.log(probs[i][word])
            sentences[:,pos] = chosen_words
        return sentences,edit_num/len(rand_list),prob,cond_prob

def CalcSentProbBatch(batched_sentences):
    seq_len = batched_sentences.shape[-1]
    sent_prob = np.array([CalcProbBatch(batched_sentences,i) for i in range(1,seq_len-1)])
    return np.sum(sent_prob,axis=0)

def CalcProbBatch(batched_sentences,i):
    masked_sentences = batched_sentences.clone()
    masked_sentences[:,i] = mask_id
    outputs = model(masked_sentences)
    probs = torch.log(F.softmax(outputs[0][:,i]/temp,dim=-1))
    return [probs[j,batched_sentence[i]].item() for j,batched_sentence in enumerate(batched_sentences)]

core_id = args[2]
os.environ["CUDA_VISIBLE_DEVICES"] = core_id
#Load the model
tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
model = BertForMaskedLM.from_pretrained('bert-large-cased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
mask_id = tokenizer.encode("[MASK]")[1:-1][0]

#Parameters
batch_size = 20
#batch_num = 10
chain_len = 10000
prob_sample = 200
temp = 1
sampling_method = 'gibbs'
sentence_id = args[1]
sent_sample = 5
edit_sample = 200

#Load sentences
with open(f'{sentence_id}.txt','r') as f:
    input_sentences = f.read().split('\n')[:-1]
#assert len(input_sentences) == batch_num
batch_num = len(input_sentences)

#Preassign arrrays
edit_rate_array = np.zeros((batch_num,batch_size,chain_len//edit_sample))
prob_array = np.zeros((batch_num,batch_size,chain_len//prob_sample))
cond_prob_array_list = []

#Run the sampling
with open(f'textfile/bert_{sampling_method}_{sentence_id}_{temp}.csv','w') as f:
    writer = csv.writer(f)
    head = ['index','initial_sentence','batch_num','iter_num']+[f'chain {k}' for k in range(batch_size)]
    writer.writerow(head)
    for i in range(batch_num):
        time1 = time.time()
        sentence = input_sentences[i]
        words = sentence.split(" ")
        words[0] = words[0].capitalize()
        words.append(".")
        init_sentence = tokenizer(" ".join(words), return_tensors="pt")["input_ids"][0].to(device)
        decoded_init_sent = tokenizer.decode(init_sentence)
        print("initial_sentence: "+decoded_init_sent)
        init_input = init_sentence.expand((batch_size,init_sentence.shape[0]))
        sents = init_input.clone()
        seq_len = sents.shape[1]
        if sampling_method == 'gibbs':
            cond_prob_array = np.zeros((batch_size,chain_len//edit_sample,seq_len-2))
            for j in range(chain_len):
                sents,edit_rate,prob,cond_prob = UniformGibbsSample(sents,writer,i,j,decoded_init_sent)
                if j%edit_sample==0:
                    edit_rate_array[i][:,j//edit_sample] = edit_rate
                    cond_prob_array[:,j//edit_sample] = cond_prob
                if j%prob_sample==0:
                    prob_array[i][:,j//prob_sample] = prob
                    print('iteration '+str(j))
                    print(prob)
        else:
            print("This code is only for Gibbs sampling.")
            exit()
        cond_prob_array_list.append(cond_prob_array)
        time2 = time.time()
        print(f'Time it took for {i}th batch: {time2-time1}')

with open(f'datafile/edit_rate_array_{sampling_method}_{sentence_id}_{temp}.pkl','wb') as f:
    pickle.dump(edit_rate_array,f)
with open(f'datafile/prob_array_{sampling_method}_{sentence_id}_{temp}.pkl','wb') as f:
    pickle.dump(prob_array,f)
with open(f'datafile/cond_prob_array_{sampling_method}_{sentence_id}_{temp}.pkl','wb') as f:
    pickle.dump(cond_prob_array_list,f)
