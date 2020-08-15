import torch
import numpy as np
import pickle
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import torch.nn.functional as F
import csv
import sys
sys.path.append('..')
args = sys.argv

def UniformGibbsSample(sentences,writer,batch_id,iter_num,decoded_init_sent):
    seq_len = sentences.shape[1]
    rand_list = np.random.permutation(seq_len-2)+1
    with torch.no_grad():
        edit_num = np.zeros(sentences.shape[0])
        if iter_num%prob_sample==0:
            prob = CalcSentProbBatch(sentences)
        index =chain_len*batch_id+iter_num
        writer.writerow([str(index),decoded_init_sent,str(batch_id),str(iter_num)]+[tokenizer.decode(sentence) for sentence in sentences])
        cond_prob = np.zeros((sentences.shape[0],len(rand_list)))
        for pos_id,pos in enumerate(rand_list):
            masked_sentences = sentences.clone()
            masked_sentences[:,pos] = mask_id
            outputs = model(masked_sentences)
            probs = F.softmax(outputs[0][:,pos]/temp,dim=-1)
            chosen_words = [np.random.choice(len(prob),p=prob.cpu().numpy()) for prob in probs]
            for i,word in enumerate(chosen_words):
                if word != sentences[i][pos]:
                    edit_num[i] += 1
                cond_prob[i][pos_id] = torch.log(probs[i][word])
                sentences[i][pos] = word
        print(edit_num)
        if iter_num%prob_sample==0:
            return sentences,edit_num/len(rand_list),prob,cond_prob
        else:
            return sentences,edit_num/len(rand_list),None,cond_prob

def UniformMHSample(sentences,writer,batch_id,iter_num,decoded_init_sent):
    seq_len = sentences.shape[1]
    rand_list = np.random.permutation(seq_len-2)+1
    with torch.no_grad():
        if iter_num%prob_sample==0:
            prob = CalcSentProbBatch(sentences)
        edit_num = np.zeros(sentences.shape[0])
        index =chain_len*batch_id+iter_num
        writer.writerow([str(index),decoded_init_sent,str(batch_id),str(iter_num)]+[tokenizer.decode(sentence) for sentence in sentences])
        cond_prob = np.zeros((sentences.shape[0],len(rand_list)))
        for pos_id,pos in enumerate(rand_list):
            old_sent_prob = CalcSentProbBatch(sentences)
            old_cond_prob = np.zeros(sentences.shape[0])
            new_cond_prob = np.zeros(sentences.shape[0])
            masked_sentences = sentences.clone()
            masked_sentences[:,pos] = mask_id
            outputs = model(masked_sentences)
            probs = F.softmax(outputs[0][:,pos]/temp,dim=-1)
            chosen_words = [np.random.choice(len(prob),p=prob.cpu().numpy()) for prob in probs]
            assert sentences.shape[0] == len(chosen_words)
            for i,word in enumerate(chosen_words):
                old_cond_prob[i] = torch.log(probs[i][sentences[i][pos]])
                new_cond_prob[i] = torch.log(probs[i][word])
                masked_sentences[i][pos] = word
            new_sent_prob = CalcSentProbBatch(masked_sentences)
            for i,word in enumerate(chosen_words):
                alpha = np.exp(new_sent_prob[i]-old_sent_prob[i]+old_cond_prob[i]-new_cond_prob[i])
                if np.random.random() < min(1,alpha):
                    if sentences[i][pos].item() != word:
                        edit_num[i] += 1
                    sentences[i][pos] = word
                    old_cond_prob[i] = torch.log(probs[i][word])
            cond_prob[:,pos_id] = old_cond_prob
        print(edit_num)
        if iter_num%prob_sample==0:
            return sentences,edit_num/len(rand_list),prob,cond_prob
        else:
            return sentences,edit_num/len(rand_list),None,cond_prob

def CalcSentProb(input_sentence):
    sent_prob = np.array([CalcProb(input_sentence,i) for i in range(1,len(input_sentence)-1)])
    return np.sum(sent_prob)

def CalcProb(input_sentence,i):
    masked_sentence = input_sentence.clone()
    masked_sentence[i] = mask_id
    outputs = model(masked_sentence.unsqueeze(0))
    probs = torch.log(F.softmax(outputs[0][0][i]/temp,dim=0))
    return probs[input_sentence[i]].item()

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


#Load the model
tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
model = BertForMaskedLM.from_pretrained('bert-large-cased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
mask_id = tokenizer.encode("[MASK]")[1:-1][0]

#Parameters
batch_size = 10
batch_num = 10
chain_len = 100
prob_sample = 5
temp = 1
sampling_method = 'metropolis'
sentence_id = 'high_low_long'

#Load sentences
with open(f'high_low_prob_sents_simple_long.txt','r') as f:
    sentences = f.read().split('\n')[:-1]
assert len(sentences) == batch_num

#Preassign arrrays
edit_rate_array = np.zeros((batch_num,batch_size,chain_len))
prob_array = np.zeros((batch_num,batch_size,chain_len//prob_sample))
cond_prob_array_list = []

#Run the sampling
with open(f'textfile/bert_{sampling_method}_{sentence_id}_{temp}.csv','w') as f:
    writer = csv.writer(f)
    head = ['index','initial_sentence','batch_num','iter_num']+[f'chain {k}' for k in range(batch_size)]
    writer.writerow(head)
    for i in range(batch_num):
        sentence = sentences[i]
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
            cond_prob_array = np.zeros((batch_size,chain_len,seq_len-2))
            for j in range(chain_len):
                print('iteration '+str(j))
                sents,edit_rate,prob,cond_prob = UniformGibbsSample(sents,writer,i,j,decoded_init_sent)
                edit_rate_array[i][:,j] = edit_rate
                cond_prob_array[:,j] = cond_prob
                if j%prob_sample==0:
                    prob_array[i][:,j//prob_sample] = prob
                    print(prob)
        elif sampling_method == 'metropolis':
            cond_prob_array = np.zeros((batch_size,chain_len,seq_len-2))
            for j in range(chain_len):
                print('iteration '+str(j))
                sents,edit_rate,prob,cond_prob = UniformMHSample(sents,writer,i,j,decoded_init_sent)
                edit_rate_array[i][:,j] = edit_rate
                cond_prob_array[:,j] = cond_prob
                if j%prob_sample==0:
                    prob_array[i][:,j//prob_sample] = prob
                    print(prob)
        cond_prob_array_list.append(cond_prob_array)

with open(f'datafile/edit_rate_array_{sampling_method}_{sentence_id}_{temp}.pkl','wb') as f:
    pickle.dump(edit_rate_array,f)
with open(f'datafile/prob_array_{sampling_method}_{sentence_id}_{temp}.pkl','wb') as f:
    pickle.dump(prob_array,f)
with open(f'datafile/cond_prob_array_{sampling_method}_{sentence_id}_{temp}.pkl','wb') as f:
    pickle.dump(cond_prob_array_list,f)
