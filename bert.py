import torch
import numpy as np
import pickle
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import torch.nn.functional as F
import csv
import sys
sys.path.append('..')
args = sys.argv

def UniformNaiveSample(sentences,writer,batch_id,iter_num,decoded_init_sent):
    seq_len = sentences.shape[1]
    rand_list = np.random.permutation(seq_len-2)+1
    with torch.no_grad():
        edit_num = np.zeros(sentences.shape[0])
        if iter_num%prob_sample==0:
            prob = CalcSentProbBatch(sentences)
        index =chain_len*batch_id+iter_num
        writer.writerow([str(index),decoded_init_sent,str(batch_id),str(iter_num)]+[tokenizer.decode(sentence) for sentence in sentences])
        for pos in rand_list:
            masked_sentences = sentences.clone()
            masked_sentences[:,pos] = mask_id
            outputs = model(masked_sentences)
            probs = F.softmax(outputs[0][:,pos]/temp,dim=-1)
            chosen_words = [np.random.choice(len(prob),p=prob.cpu().numpy()) for prob in probs]
            for i,word in enumerate(chosen_words):
                if word != sentences[i][pos]:
                    edit_num[i] += 1
                sentences[i][pos] = word
        if iter_num%prob_sample==0:
            return sentences,edit_num/len(rand_list),prob
        else:
            return sentences,edit_num/len(rand_list),None

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



tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
model = BertForMaskedLM.from_pretrained('bert-large-cased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
mask_id = tokenizer.encode("[MASK]")[1:-1][0]
batch_size = 50
batch_num = 5
chain_len = 500
prob_sample = 10
temp = 0.5

sentence_id = 39
sentence = "He could see that they were determined to turn back"
words = sentence.split(" ")
words.append(".")
init_sentence = tokenizer(" ".join(words), return_tensors="pt")["input_ids"][0].to(device)
decoded_init_sent = tokenizer.decode(init_sentence)
print("initial_sentence: "+decoded_init_sent)
init_input = init_sentence.expand((batch_size,init_sentence.shape[0]))

edit_rate_array = np.zeros((batch_num,batch_size,chain_len))
prob_array = np.zeros((batch_num,batch_size,chain_len//prob_sample))
with open(f'textfile/bert_{sentence_id}_{temp}.csv','w') as f:
    writer = csv.writer(f)
    head = ['index','initial_sentence','batch_num','iter_num']+[f'chain {k}' for k in range(batch_size)]
    writer.writerow(head)
    for i in range(batch_num):
        sents = init_input.clone()
        for j in range(chain_len):
            print('iteration '+str(j))
            sents,edit_rate,prob = UniformNaiveSample(sents,writer,i,j,decoded_init_sent)
            edit_rate_array[i][:,j] = edit_rate
            if j%prob_sample==0:
                prob_array[i][:,j//prob_sample] = prob
                print(prob)

with open(f'datafile/edit_rate_array_{sentence_id}_{temp}.pkl','wb') as f:
    pickle.dump(edit_rate_array,f)
with open(f'datafile/prob_array_{sentence_id}_{temp}.pkl','wb') as f:
    pickle.dump(prob_array,f)
