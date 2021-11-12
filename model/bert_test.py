# export WIKI_PATH='YOUR PATH TO WIKICORPUS'
# python bert.py --sentence_id input --core_id 1 --num_tokens 21
import torch
import numpy as np
import pickle
from transformers import BertTokenizer, BertForMaskedLM
import torch.nn.functional as F
import csv
import time
import sys
import os
import argparse
import spacy
import math
import glob

class Writer():
    def __init__(self, args, save_file) :
        self.batch_size = args.batch_size
        self.chain_len = args.chain_len
        self.save_file = save_file
        self.init_output_csv()

    def init_output_csv(self, header=None):
        header = ['sentence_num','chain_num','iter_num',
                  'initial_sentence', 'sentence', 'prob',
                  'edit_rate', 'step','switch','accept_rate']
        with open(self.save_file, "w") as writeFile:
            csv.writer(writeFile).writerow(header)

    def reset(self, sentence_num, init_sent) :
        self.init_sent = init_sent
        self.sentence_num = sentence_num

    def write(self, iter_num, sentences, scores, edit_rate, step, switch, accept_rate):
        with open(self.save_file, "a") as writeFile:
            csvwriter = csv.writer(writeFile)
            decoded_sentences = [str(tokenizer.decode(sentence)) for sentence in sentences]
            if iter_num%100==0:
                print(iter_num)
                print(decoded_sentences)
                print(scores)
                print(edit_rate)
                print(accept_rate)
            for row_id, sentence in enumerate(decoded_sentences) :
                csvwriter.writerow([
                    self.sentence_num, row_id, iter_num,
                    self.init_sent, sentence, scores[row_id],
                    edit_rate[row_id].item(), step, switch, accept_rate[row_id].item()
                ])

class UniformGibbs():
    def __init__(self, sentences, temp, model, mask_id, device, sweep_order):
        self.sentences = sentences
        self.temp = temp
        self.model = model
        self.mask_id = mask_id
        self.device = device
        self.edit_rate = torch.zeros(self.sentences.shape[0],device=self.device)
        self.accept_rate = torch.ones(self.sentences.shape[0],device=self.device) # always accept gibbs samples
        self.sweep_order = sweep_order

    def get_rand_list(self,seq_len):
        # exclude first/last tokens (CLS/SEP) from positions
        if self.sweep_order=='ascend':
            rand_list = torch.arange(seq_len-2,device=self.device)+1
        elif self.sweep_order=='descend':
            rand_list = torch.arange(seq_len-2,0,-1,device=self.device)
        elif self.sweep_order=='random_sweep':
            rand_list = torch.randperm(seq_len-2,device=self.device)+1
        elif self.sweep_order=='random':
            rand_list = torch.randint(seq_len-2,size=(seq_len-2,),device=self.device)+1
        else:
            print('Invalid sweep_order')
        return rand_list

    @torch.no_grad()
    def step(self, iter_num, pos):
        probs = self.mask_prob(pos,self.sentences,temp=self.temp)
        sentences, edit_loc = self.sample_words(probs, pos, self.sentences)
        return sentences, edit_loc.float()

    @torch.no_grad()
    def sweep(self, iter_num):
        seq_len = self.sentences.shape[1]
        rand_list = self.get_rand_list(seq_len)
        edit_locs = torch.zeros(size=(self.sentences.shape[0],len(rand_list)),device=self.device)
        for pos_id,pos in enumerate(rand_list):
            self.sentences, edit_locs[:, pos_id] = self.step(iter_num, pos)
        self.edit_rate = torch.mean(edit_locs, axis=1)

    def sample_words(self, probs, pos, sentences):
        chosen_words = torch.multinomial(torch.exp(probs), num_samples=1).squeeze(dim=-1)
        new_sentences = sentences.clone()
        new_sentences[:, pos] = chosen_words
        edit_loc = new_sentences[:, pos]!=sentences[:, pos]
        return new_sentences, edit_loc

    def mask_prob(self, position, sentences, temp=1):
        masked_sentences = sentences.clone()
        masked_sentences[:, position] = self.mask_id
        logits = self.model(masked_sentences)[0]
        return F.log_softmax(logits[:, position] / temp, dim = -1)

    @torch.no_grad()
    def get_total_score(self, sentences):
        sent_probs = torch.zeros_like(sentences).float()
        # Calculate masked probabilities for the actual words in the sentences
        for j in range(1,sentences.shape[1]-1):
            probs = self.mask_prob(j,sentences)
            for i in range(sentences.shape[0]):
                # Look up probability of the actual word at this position
                sent_probs[i, j] = probs[i, sentences[i, j]]
        # Sum log probs for each sentence in batch
        return torch.sum(sent_probs, axis=1)

class MultiSiteMH():
    def __init__(self, sentences, temp, model, mask_id, num_masks, device, sweep_order, adjacent_block=False):
        self.sentences = sentences
        self.temp = temp
        self.model = model
        self.mask_id = mask_id
        self.num_masks = num_masks
        self.device = device
        self.edit_rate = torch.zeros(self.sentences.shape[0],device=self.device)
        self.accept_rate = torch.zeros(self.sentences.shape[0],device=self.device)
        self.sweep_order = sweep_order
        self.adjacent_block = adjacent_block
        self.total_score = self.get_total_score(self.sentences)

    def get_rand_list(self,seq_len):
        # exclude first/last tokens (CLS/SEP) from positions
        if self.sweep_order=='ascend':
            rand_list = np.arange(seq_len-2)+1
        elif self.sweep_order=='descend':
            rand_list = np.arange(seq_len-2)[::-1]+1
        elif self.sweep_order=='random_sweep':
            rand_list = np.random.permutation(seq_len-2)+1
        elif self.sweep_order=='random':
            rand_list = np.random.randint(seq_len-2,size=seq_len-2)+1
        else:
            print('Invalid sweep_order')
        return rand_list

    def get_mask_pos_list(self,rand_list,seq_len):
        if self.adjacent_block:
            mask_pos_list = torch.tensor([[(pos-1+i)%(seq_len-2)+1 for i in range(self.num_masks)] for pos in rand_list],device=self.device)
        else:
            mask_pos_list = torch.tensor([[pos]+list(np.random.choice([i for i in rand_list if i!=pos],
                                                                        size=self.num_masks-1,replace=False))
                                                                        for pos in rand_list],device=self.device)
        return mask_pos_list

    @torch.no_grad()
    def step(self, iter_num, pos):
        probs = self.mask_prob(pos,self.sentences,temp=self.temp)
        sentences, edit_loc, accept_rate = self.sample_words(probs, pos, self.sentences)
        edit_rate = edit_loc.mean(axis=1)
        return sentences, edit_rate, accept_rate

    @torch.no_grad()
    def sweep(self, iter_num):
        seq_len = self.sentences.shape[1]
        rand_list = self.get_rand_list(seq_len)
        mask_pos_list = self.get_mask_pos_list(rand_list,seq_len)
        edit_locs = torch.zeros(size=(self.sentences.shape[0],len(rand_list)),device=self.device)
        accept_locs = torch.zeros(size=(self.sentences.shape[0],len(rand_list)),device=self.device)
        for pos_id, pos in enumerate(mask_pos_list):
            self.sentences, edit_locs[:, pos_id], accept_locs[:, pos_id] = self.step(iter_num, pos)
        self.edit_rate = torch.mean(edit_locs, axis=1)
        self.accept_rate = torch.mean(accept_locs, axis=1)

    def sample_words(self, probs, pos, sentences):
        #old_score = self.get_total_score(sentences)
        #assert torch.all(self.total_score==old_score)
        old_score = self.total_score.clone()
        old_words = sentences[:,pos]
        #Propose a set of words
        new_words = torch.tensor([list(torch.multinomial(prob, num_samples=1).squeeze(dim=-1)) for prob in torch.exp(probs)],device=self.device)
        new_sentences = sentences.clone()
        new_sentences[:,pos] = new_words
        new_score = self.get_total_score(new_sentences)
        fwd_prob = torch.tensor([prob.index_select(dim=-1,index=word_list).diagonal().sum().item()\
                                for word_list,prob in zip(new_words,probs)],device=self.device)
        bck_prob = torch.tensor([prob.index_select(dim=-1,index=word_list).diagonal().sum().item()\
                                for word_list,prob in zip(old_words,probs)],device=self.device)
        alpha = torch.exp(new_score - old_score + bck_prob - fwd_prob)
        alpha[alpha>1] = 1
        accept = torch.rand(sentences.shape[0],device=self.device)<alpha
        chosen_words = old_words.clone()
        chosen_words[accept,:] = new_words[accept,:]
        chosen_sentences = sentences.clone()
        chosen_sentences[:,pos] = chosen_words
        edit_rate = chosen_sentences[:,pos]!=sentences[:,pos]
        chosen_score = old_score.clone()
        chosen_score[accept] = new_score[accept]
        self.total_score = chosen_score
        return chosen_sentences, edit_rate.float(), accept.float()

    def mask_prob(self, position, sentences, temp=1):
        masked_sentences = sentences.clone()
        masked_sentences[:, position] = self.mask_id
        logits = self.model(masked_sentences)[0]
        return F.log_softmax(logits[:, position] / temp, dim = -1)

    @torch.no_grad()
    def get_total_score(self, sentences):
        sent_probs = torch.zeros_like(sentences).float()
        # Calculate masked probabilities for the actual words in the sentences
        for j in range(1,sentences.shape[1]-1):
            probs = self.mask_prob(j,sentences)
            for i in range(sentences.shape[0]):
                # Look up probability of the actual word at this position
                sent_probs[i, j] = probs[i, sentences[i, j]]
        # Sum log probs for each sentence in batch
        return torch.sum(sent_probs, axis=1)

@torch.no_grad()
def run_chains(args) :
    # Load sentences
    with open(f'initial_sentences/{args.num_tokens}Tokens/{args.sentence_id}.txt','r') as f:
        input_sentences = f.read().split('\n')[:-1]
        batch_num = len(input_sentences)

    # Load wiki sentences (when running 'gibbs_mixture' these will be used as new initial sentences)
    if args.sampling_method=='gibbs_mixture' and ~args.mask_initialization:
        files = glob.glob(f'{os.environ.get("WIKI_PATH")}/Samples/CleanedSamples/{args.num_tokens}TokenSents/*.csv')
        assert len(files)==1
        with open(files[0],'r') as f:
            reader = csv.reader(f)
            file = [row for row in reader]
        head = file[0]
        text = file[1:]
        wiki_text = [row[head.index('sentence')] for row in text if int(row[head.index('num_tokens_flag')])==1 and int(row[head.index('sent_num_flag')==1])]
        wiki_samples = tokenizer(wiki_text,return_tensors="pt")["input_ids"]
        assert wiki_samples.shape[1]==args.num_tokens

    # Set up output dirs/files
    os.makedirs(f'BertData/{args.num_tokens}TokenSents/textfile/{args.model_name}/{args.batch_size}_{args.chain_len}/', exist_ok=True)

    if args.sampling_method=='gibbs_mixture':
        f = f'BertData/{args.num_tokens}TokenSents/textfile/{args.model_name}/{args.batch_size}_{args.chain_len}/'\
        +f'bert_new_{args.sampling_method}_{args.sweep_order}_{args.epsilon}{args.mask_init_id}_{args.sentence_id}_{args.temp}.csv'
    elif args.adjacent_block:
        f = f'BertData/{args.num_tokens}TokenSents/textfile/{args.model_name}/{args.batch_size}_{args.chain_len}/'\
        +f'bert_new_{args.sampling_method}_adjacent_{args.sweep_order}{args.mask_init_id}_{args.sentence_id}_{args.temp}.csv'
    else:
        f = f'BertData/{args.num_tokens}TokenSents/textfile/{args.model_name}/{args.batch_size}_{args.chain_len}/'\
        +f'bert_new_{args.sampling_method}_{args.sweep_order}{args.mask_init_id}_{args.sentence_id}_{args.temp}.csv'
    writer = Writer(args, f)

    for i, input_sentence in enumerate(input_sentences):
        print(f'Beginning batch {i}')
        start = time.time()

        # set up the input
        words = input_sentence.capitalize()
        tokenized_sentence = tokenizer(words, return_tensors="pt")
        if args.mask_initialization:
            init_input = (torch.tensor([args.cls_id]+[args.mask_id for _ in range(args.num_tokens-2)]+[args.sep_id])
                          .to(args.device)
                          .expand((args.batch_size, -1)))
        else:
            init_input = (tokenized_sentence["input_ids"][0]
                          .to(args.device)
                          .expand((args.batch_size, -1)))#, init_sentence.shape[0])))
        assert init_input.shape[1] == args.num_tokens, 'number of tokens in the initial sentence does not match the num_tokens argument'
        # reset writer
        writer.reset(i, words)

        # set up the sampler
        if 'gibbs' in args.sampling_method:
            sampler = UniformGibbs(init_input, args.temp, model, args.mask_id, args.device, args.sweep_order)
        elif 'mh' in args.sampling_method:
            sampler = MultiSiteMH(init_input, args.temp, model, args.mask_id, args.num_masks, args.device, args.sweep_order, args.adjacent_block)

        switch = 0
        num_switches = 0
        for iter_num in range(args.chain_len):
            # write out all steps for iter_num<100
            if iter_num<1000:
                seq_len = sampler.sentences.shape[1]
                if 'gibbs' in args.sampling_method:
                    # exclude first/last tokens (CLS/SEP) from positions
                    rand_list = sampler.get_rand_list(seq_len)
                    for pos_id,pos in enumerate(rand_list):
                        writer.write(iter_num, sampler.sentences,
                                     sampler.get_total_score(sampler.sentences).cpu().detach().numpy(),
                                     sampler.edit_rate,pos_id,0,sampler.accept_rate)
                        sampler.sentences, sampler.edit_rate = sampler.step(iter_num, pos)
                elif 'mh' in args.sampling_method:
                    # exclude first/last tokens (CLS/SEP) from positions
                    start = time.time()
                    rand_list = sampler.get_rand_list(seq_len)
                    mask_pos_list = sampler.get_mask_pos_list(rand_list,seq_len)
                    for pos_id,pos in enumerate(mask_pos_list):
                        writer.write(iter_num, sampler.sentences,
                                     sampler.total_score.cpu().detach().numpy(),
                                     sampler.edit_rate,pos_id,0,sampler.accept_rate)
                        sampler.sentences, sampler.edit_rate, sampler.accept_rate = sampler.step(iter_num, pos)
                    print(time.time()-start)
                    exit()

            else:
                if iter_num % args.sent_sample == 0:
                    if 'gibbs' in args.sampling_method:
                        writer.write(iter_num, sampler.sentences,
                                     sampler.get_total_score(sampler.sentences).cpu().detach().numpy(),
                                     sampler.edit_rate,sampler.sentences.shape[1]-3,switch,sampler.accept_rate)
                    elif 'mh' in args.sampling_method:
                        writer.write(iter_num, sampler.sentences,
                                     sampler.total_score.cpu().detach().numpy(),
                                     sampler.edit_rate,sampler.sentences.shape[1]-3,switch,sampler.accept_rate)
                    switch = 0
                sampler.sweep(iter_num)
                if args.sampling_method=='gibbs_mixture' and iter_num>=1000 and torch.rand(1)<args.epsilon:
                    switch = 1
                    num_switches += 1

                    if args.mask_initialization:
                        new_init_input = (torch.tensor([args.cls_id]+[args.mask_id for _ in range(args.num_tokens-2)]+[args.sep_id])
                                        .to(args.device)
                                        .expand((args.batch_size, -1)))
                    else:
                        # Choose random wikipedia samples
                        sampled_ids = torch.randperm(wiki_samples.shape[0],device=args.device)[:args.batch_size]
                        new_init_input = wiki_samples[sampled_ids].to(args.device)

                    sampler = UniformGibbs(new_init_input, args.temp, model, args.mask_id, args.device, args.sweep_order)
                    for _ in range(100):
                        sampler.sweep(iter_num)
        print(f'# of switches: {num_switches}')
        print(f'Time it took for {i}th batch: {time.time()-start}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentence_id', type = str, required = True)
    parser.add_argument('--core_id', type = str, required = True)
    parser.add_argument('--num_tokens',type=int, required = True,
                        help='number of tokens including special tokens')
    parser.add_argument('--model_name', type = str, default = 'bert-base-uncased')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='number of sentences to maintain')
    parser.add_argument('--chain_len', type=int, default = 10000,
                        help='number of samples')
    parser.add_argument('--temp', type = float, default = 1,
                        help='softmax temperature')
    parser.add_argument('--sent_sample', type=int, default = 5,
                        help='frequency of recording sentences')
    parser.add_argument('--sampling_method', type=str, required=True,
                        help='kind of sampling to do; options include "gibbs", "gibbs_mixture" and "mh_{num_masks}"')
    parser.add_argument('--epsilon', type=float,
                        help='epsilon when using gibbs_mixture')
    parser.add_argument('--sweep_order', type=str, required = True,
                        choices=['ascend','descend','random_sweep','random'])
    parser.add_argument('--adjacent_block', dest='adjacent_block', action='store_true', default=False)
    parser.add_argument('--mask_initialization', dest='mask_initialization', action='store_true', default=False)
    args = parser.parse_args()

    if 'mh' in args.sampling_method:
        args.num_masks = int(args.sampling_method.split('_')[1])

    if 'gibbs' in args.sampling_method:
        assert args.temp==1, 'temp is only for MH'

    if args.adjacent_block:
        assert 'mh' in args.sampling_method, 'adjacent_block is only for MH'

    if args.mask_initialization:
        args.mask_init_id = '_mask_init'
    else:
        args.mask_init_id = ''

    print('running with args', args)

    #os.environ["CUDA_VISIBLE_DEVICES"] = args.core_id
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertForMaskedLM.from_pretrained(args.model_name)

    if torch.cuda.is_available():
        args.device = torch.device("cuda", index=int(args.core_id))
    else:
        args.device = torch.device("cpu")
    model.to(args.device)
    model.eval()
    args.mask_id = tokenizer.encode("[MASK]")[1:-1][0]
    args.cls_id = tokenizer.encode("[MASK]")[0]
    args.sep_id = tokenizer.encode("[MASK]")[-1]

    # launch chains
    run_chains(args)
