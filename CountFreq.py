# export WIKI_PATH='YOUR PATH TO WIKICORPUS'
# export BOOK_PATH='YOUR PATH TO BOOKCORPUS'
# python bert.py --sentence_id input --core_id 1 --num_tokens 21
import torch
import numpy as np
import pickle
import csv
import sys
from spacy.symbols import ORTH
import time
from multiprocessing import Pool
import os
import argparse
import glob
import math

def TakeOutFuncTokens(sentence):
    '''
        Take out special tokens
    '''
    return sentence.replace('[CLS]','').replace('[SEP]','').strip()

def LoadCorpus(args,file_name,text_path,nlp,burn_in=True):
    '''
        Load either wikipedia or bert-generated sentences
        For bert, the first 1000 is for the burn-in period
    '''
    if args.corpus in ['wiki','book']:
        with open(f'{text_path}{file_name}.txt','r') as f:
            sentences = f.read().split('\n')[:-1]
    elif args.corpus == 'bert':
        with open(f'{file_name}.csv','r') as f:
            reader = csv.reader(f)
            file = [row for row in reader]
            head = file[0]
            text = file[1:]
        if burn_in:
            sentences = [TakeOutFuncTokens(row[head.index(f'sentence')]) for row in text if int(row[head.index('iter_num')]) >= 1000 and int(row[head.index('iter_num')])%args.sent_sample==0]
        else:
            sentences = [TakeOutFuncTokens(row[head.index(f'sentence')]) for row in text if int(row[head.index('iter_num')])%args.sent_sample==0]
    return sentences

def ExtractFreqNew(sentences,tokenizer,sent_tokenize,nlp,args,verbose=False):
    Freq = {}
    ShortDep = ""
    sent_num = 0
    for sent in sentences:
        if len(sent_tokenize(sent))==1 and len(tokenizer(sent)['input_ids'])==args.num_tokens and '|' not in sent and sent[-1] not in [';',':']:
            line = nlp(sent)
            sent_num += 1
            if args.metric == 'dep_dist':
                total_dist = np.array([abs(token_pos-token.head.i) for token_pos,token in enumerate(line)]).sum()
                if total_dist not in Freq:
                    Freq[total_dist] = 0
                Freq[total_dist] += 1
                if total_dist == 10:
                    ShortDep += line.text+'\n'
            else:
                for token in line:
                    if args.metric in ['vocab', 'pos', 'tag', 'dep']:
                        if args.metric == 'vocab':
                            word = token.text
                        elif args.metric == 'pos':
                            word = token.pos_
                        elif args.metric == 'tag':
                            word = token.tag_
                        elif args.metric == 'dep':
                            word = token.dep_

                        if word not in Freq:
                            Freq[word] = 0
                        Freq[word] += 1
        elif verbose:
            print(sent)
            print(sent_tokenize(sent))
            print(len(tokenizer(sent)['input_ids']))
    return Freq, ShortDep, sent_num

def WriteOutFreq(args,file_name,text_path,nlp):
    '''
        Extract frequency of the specified 'metric' and return dictionary
    '''
    sentences = LoadCorpus(args,file_name,text_path,nlp)
    batched_sentences = batchify(sentences,1000)
    arg = [(batch,bert_tokenizer,sent_tokenize,nlp,args) for batch in batched_sentences]
    with Pool(processes=100) as p:
        Results = p.starmap(ExtractFreqNew,arg)
    FreqList = []
    ShortDepList = []
    sent_num_list = []
    for line in Results:
        FreqList.append(line[0])
        ShortDepList.append(line[1])
        sent_num_list.append(line[2])
    FreqDictAll = UnifyDict(FreqList,args)
    print(f'# of sentences for {file_name}: {np.sum(sent_num_list)}')
    return [FreqDictAll,ShortDepList]

def TokenizerSetUpNew():
    import spacy
    nlp = spacy.load('en_core_web_lg')
    nlp.tokenizer.add_special_case("[UNK]",[{spacy.symbols.ORTH: "[UNK]"}])
    return nlp

def batchify(data,batch_size):
    batch_num = math.ceil(len(data)/batch_size)
    return [data[batch_size*i:batch_size*(i+1)] for i in range(batch_num)]

def UnifyDict(DictList,args):
    FreqDictAll = {}
    for Dict in DictList:
        for word in Dict:
            if word not in FreqDictAll:
                FreqDictAll[word] = 0
            FreqDictAll[word] += Dict[word]
    return FreqDictAll

if __name__ == '__main__':
    ##Organize arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type = str, required = True, choices=['wiki', 'bert', 'book'])
    parser.add_argument('--metric', type = str, required = True, choices=['vocab', 'pos', 'tag', 'dep', 'dep_dist'])
    parser.add_argument('--num_tokens', type = int, required = True)
    parser.add_argument('--model_name', type = str, default = 'bert-base-uncased')
    #The rest are only for bert
    parser.add_argument('--batch_size', type = int)
    parser.add_argument('--chain_len', type = int)
    parser.add_argument('--temp', type = float, default = 1,
                        help='softmax temperature')
    parser.add_argument('--sent_sample', type = int)
    parser.add_argument('--sampling_method', type=str,
                        help='kind of sampling to do; options include "gibbs", "gibbs_mixture" and "mh_{num_masks}"')
    parser.add_argument('--epsilon', type=float,
                        help='epsilon when using gibbs_mixture')
    parser.add_argument('--sweep_order', type=str,
                        choices=['ascend','descend','random_sweep','random'])
    parser.add_argument('--adjacent_block', dest='adjacent_block', action='store_true', default=False)
    parser.add_argument('--mask_initialization', dest='mask_initialization', action='store_true', default=False)
    args = parser.parse_args()

    if args.corpus=='bert':
        assert args.sweep_order is not None
        if 'mh' in args.sampling_method:
            args.num_masks = int(args.sampling_method.split('_')[-1])

        if 'gibbs' in args.sampling_method:
            assert args.temp==1, 'temp is only for MH'

        if args.adjacent_block:
            assert 'mh' in args.sampling_method, 'adjacent_block is only for MH'

        if args.mask_initialization:
            args.mask_init_id = '_mask_init'
        else:
            args.mask_init_id = ''

    print('running with args', args)

    from transformers import BertTokenizer
    bert_tokenizer = BertTokenizer.from_pretrained(args.model_name)
    from nltk.tokenize import sent_tokenize

    ##Specify proper paths and gather file names
    if args.corpus == 'bert':
        if 'mixture' in args.sampling_method:
            text_path = f'BertData/{args.num_tokens}TokenSents/textfile/{args.model_name}/{args.batch_size}_{args.chain_len}/'\
            +f'bert_new_{args.sampling_method}_{args.sweep_order}_{args.epsilon}{args.mask_init_id}_input_*{args.temp}.csv'
            data_path = f'BertData/{args.num_tokens}TokenSents/datafile/{args.model_name}/'\
            +f'{args.batch_size}_{args.chain_len}_{args.sent_sample}_{args.sampling_method}_{args.sweep_order}{args.mask_init_id}_{args.temp}_{args.epsilon}/'
        elif args.adjacent_block:
            text_path = f'BertData/{args.num_tokens}TokenSents/textfile/{args.model_name}/{args.batch_size}_{args.chain_len}/'\
            +f'bert_new_{args.sampling_method}_adjacent_{args.sweep_order}{args.mask_init_id}_input_*{args.temp}.csv'
            data_path = f'BertData/{args.num_tokens}TokenSents/datafile/{args.model_name}/'\
            +f'{args.batch_size}_{args.chain_len}_{args.sent_sample}_{args.sampling_method}_{args.sweep_order}{args.mask_init_id}_{args.temp}_adjacent/'
        else:
            text_path = f'BertData/{args.num_tokens}TokenSents/textfile/{args.model_name}/{args.batch_size}_{args.chain_len}/'\
            +f'bert_new_{args.sampling_method}_{args.sweep_order}{args.mask_init_id}_input_*{args.temp}.csv'
            data_path = f'BertData/{args.num_tokens}TokenSents/datafile/{args.model_name}/'\
            +f'{args.batch_size}_{args.chain_len}_{args.sent_sample}_{args.sampling_method}_{args.sweep_order}{args.mask_init_id}_{args.temp}/'

        os.makedirs(data_path,exist_ok=True)
        files = [file_name.replace('.csv','') for file_name in glob.glob(f'{text_path}')]

    elif args.corpus == 'wiki':
        text_path = f'{os.environ.get("WIKI_PATH")}/TokenSents/{args.num_tokens}TokenSents/textfile/'
        data_path = f'{os.environ.get("WIKI_PATH")}/TokenSents/{args.num_tokens}TokenSents/datafile/'
        os.makedirs(data_path,exist_ok=True)
        files = [file_name.replace(f'{text_path}','').replace('.txt','') for file_name in glob.glob(f'{text_path}*.txt')]

    elif args.corpus == 'book':
        text_path = f'{os.environ.get("BOOK_PATH")}/TokenSents/{args.num_tokens}TokenSents/textfile/'
        data_path = f'{os.environ.get("BOOK_PATH")}/TokenSents/{args.num_tokens}TokenSents/datafile/'
        os.makedirs(data_path,exist_ok=True)
        files = [file_name.replace(f'{text_path}','').replace('.txt','') for file_name in glob.glob(f'{text_path}*.txt')]

    print(f'Running {files}')
    assert len(files)==1

    ##Create output dirs when necessary
    os.makedirs(f'{data_path}CountFiles/',exist_ok=True)
    os.makedirs(f'{data_path}ShortDepSents/',exist_ok=True)

    ##Set up the spacy tokenizer
    nlp = TokenizerSetUpNew()

    [FreqDictAll,ShortDepSent] = WriteOutFreq(args,files[0],text_path,nlp)

    ##Write out
    with open(f'{data_path}{args.metric.upper()}FreqAll.pkl','wb') as f:
        pickle.dump(FreqDictAll,f)
    if args.metric=='dep_dist':
        with open(f'{data_path}ShortDepSents/ShortDepSents.txt','w') as f:
            for sentence in ShortDepSent:
                f.write(sentence)
