# export WIKI_PATH='YOUR PATH TO WIKICORPUS'
# export BOOK_PATH='YOUR PATH TO BOOKCORPUS'
import torch
import numpy as np
import pickle
import csv
from multiprocessing import Pool
import os
import argparse
import glob
import time
import matplotlib.pyplot as plt
import math
from bert_new import UniformGibbs

def CalcSentProb(folder_path,text_path,args):
    folder_name = folder_path.replace(text_path,'').replace('.txt','')
    print(f'Processing {folder_name}')
    with open(folder_path,'r') as f:
        text = f.read().split('\n')[:-1]
    batch_num = math.ceil(len(text)/args.batch_size)
    batched_text_list = [text[args.batch_size*i:args.batch_size*(i+1)] for i in range(batch_num)]
    probs_tensor = torch.zeros(len(text)).detach()
    for batch_id, batched_text in enumerate(batched_text_list):
        input_sent = bert_tokenizer(batched_text,return_tensors="pt")["input_ids"].to(args.device)
        assert input_sent.shape[1] == args.num_tokens
        sampler = UniformGibbs(input_sent, args.temp, model, args.mask_id, args.device, args.sweep_order)
        probs = sampler.get_total_score(sampler.sentences)
        probs_tensor[args.batch_size*batch_id:args.batch_size*(batch_id+1)] = probs
    probs_list = list(probs_tensor)
    assert len(probs_list) == len(text)
    out_file = f'{text_path}SentProbs/{folder_name}.csv'
    if f'{folder_name}.csv' in os.listdir(f'{text_path}SentProbs/'):
        with open(out_file,'r') as f:
            reader = csv.reader(f)
            file = [row for row in reader]
            old_head = file[0]
            old_text = file[1:]
        assert len(old_text) == len(text) and len(old_text) == len(probs_list)
        with open(out_file,'w') as f:
            writer = csv.writer(f)
            if f'prob_{args.temp}' not in old_head:
                new_head = old_head + [f'prob_{args.temp}']
                writer.writerow(new_head)
                for line,new_sentence,prob in zip(old_text,text,probs_list):
                    assert line[old_head.index('sentence')] == new_sentence
                    writer.writerow(line+[f'{prob}'])
            else:
                writer.writerow(old_head)
                for line,new_sentence,prob in zip(old_text,text,probs_list):
                    assert line[old_head.index('sentence')] == new_sentence
                    line[old_head.index(f'prob_{args.temp}')] = f'{prob}'
                    writer.writerow(line)
    else:
        with open(out_file,'w') as f:
            writer = csv.writer(f)
            head = ['sent_id','sentence',f'prob_{args.temp}']
            writer.writerow(head)
            for sent_id,sentence,prob in zip(np.arange(len(text)),text,probs_list):
                writer.writerow([f'{sent_id}',sentence,f'{prob}'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, required=True, choices=['wiki','book'])
    parser.add_argument('--core_id', type = str, required=True)
    parser.add_argument('--model', type = str, default='bert-base-uncased')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='number of sentences to maintain')
    parser.add_argument('--temp', type = float, default=1,
                        help='softmax temperature')
    parser.add_argument('--num_tokens',type=int, required = True,
                        help='number of tokens including special tokens')
    args = parser.parse_args()

    from transformers import BertTokenizer,BertForMaskedLM
    bert_tokenizer = BertTokenizer.from_pretrained(args.model)
    model = BertForMaskedLM.from_pretrained(args.model)

    if torch.cuda.is_available():
        args.device = torch.device("cuda", index=int(args.core_id))
    else:
        args.device = torch.device("cpu")
    model.to(args.device)
    model.eval()
    args.mask_id = bert_tokenizer.encode("[MASK]")[1:-1][0]

    args.sweep_order = 'random'

    if args.corpus=='wiki':
        text_path = f'{os.environ.get("WIKI_PATH")}/TokenSents/{args.num_tokens}TokenSents/textfile/'
    elif args.corpus=='book':
        text_path = f'{os.environ.get("BOOK_PATH")}/TokenSents/{args.num_tokens}TokenSents/textfile/'
    os.makedirs(f'{text_path}SentProbs',exist_ok=True)
    folder_path_list = glob.glob(f'{text_path}*.txt')
    for folder_path in folder_path_list:
        start = time.time()
        CalcSentProb(folder_path,text_path,args)
        print(f'Time it took for {folder_path}: {time.time()-start}')
