import torch
import numpy as np
import pickle
from multiprocessing import Pool
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import sys
sys.path.append('..')
args = sys.argv

def NonuniformReplacement(tokenized_sentence):
    rand_list = np.random.permutation(len(tokenized_sentence)-2)+1
    print(rand_list)
    with torch.no_grad():
        for id in rand_list:
            masked_sentence = tokenized_sentence.copy()
            masked_sentence[id] = mask_id
            #print(tokenizer.decode(tokenized_sentence))
            input_tensor = torch.tensor([masked_sentence])
            outputs = model(input_tensor)
            if torch.argmax(outputs[0][0][id]) != tokenized_sentence[id]:
                tokenized_sentence[id] = torch.argmax(outputs[0][0][id])
                print(tokenizer.decode(tokenized_sentence))
                return tokenized_sentence,True
    print("Replacement Converged")
    print(tokenizer.decode(tokenized_sentence))
    return [],False

def UniformReplacement(tokenized_sentence):
    rand_list = np.random.permutation(len(tokenized_sentence)-2)+1
    print(rand_list)
    with torch.no_grad():
        for id in rand_list:
            tokenized_sentence[id] = mask_id
            print(tokenizer.decode(tokenized_sentence))
            input_tensor = torch.tensor([tokenized_sentence])
            outputs = model(input_tensor)
            tokenized_sentence[id] = torch.argmax(outputs[0][0][id])
            print(tokenizer.decode(tokenized_sentence))
    return tokenized_sentence

tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
model = BertForMaskedLM.from_pretrained('bert-large-cased')
model.eval()
mask_id = tokenizer.encode("[MASK]")[1:-1][0]

sentence = "Linda took him the meal"
words = sentence.split(" ")
words.append(".")
tokenized_sentence = tokenizer.encode(" ".join(words))
print(tokenizer.decode(tokenized_sentence))
if args[1] == 'non-uniform':
    for i in range(1):
        tokenized_sentence = tokenizer.encode(" ".join(words))
        flag = True
        while flag:
            tokenized_sentence,flag = NonuniformReplacement(tokenized_sentence)
if args[1] == 'uniform':
    for i in range(10):
        tokenized_sentence = UniformReplacement(tokenized_sentence)

