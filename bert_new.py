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
                  'edit_rate', 'substep','switch']
        with open(self.save_file, "w") as writeFile:
            csv.writer(writeFile).writerow(header)

    def reset(self, sentence_num, init_sent) :
        self.init_sent = init_sent
        self.sentence_num = sentence_num

    def write(self, iter_num, sentences, scores, edit_rate, substep, switch):
        with open(self.save_file, "a") as writeFile:
            csvwriter = csv.writer(writeFile)
            decoded_sentences = [str(tokenizer.decode(sentence)) for sentence in sentences]
            if iter_num%100==0:
                print(iter_num)
                print(decoded_sentences)
                print(scores)
                print(edit_rate)
            for row_id, sentence in enumerate(decoded_sentences) :
                csvwriter.writerow([
                    self.sentence_num, row_id, iter_num,
                    self.init_sent, sentence, scores[row_id],
                    edit_rate[row_id].item(), substep, switch
                ])

class UniformGibbs():
    def __init__(self, sentences, temp, model, mask_id):
        self.sentences = sentences
        self.temp = temp
        self.model = model
        self.mask_id = mask_id
        self.edit_rate = torch.zeros(self.sentences.shape[0]).to(args.device)

    @torch.no_grad()
    def substep(self, iter_num, pos):
        probs = self.mask_prob(pos)
        sentences, edit_loc = self.sample_words(probs, pos, self.sentences)
        return sentences, edit_loc.float()

    @torch.no_grad()
    def step(self, iter_num):
        seq_len = self.sentences.shape[1]
        # exclude first/last tokens (CLS/SEP) from positions
        rand_list = (torch.randperm(seq_len-2)+1).to(args.device)
        edit_locs = torch.zeros((self.sentences.shape[0],len(rand_list))).to(args.device)
        for pos_id,pos in enumerate(rand_list):
            self.sentences, edit_locs[:, pos_id] = self.substep(iter_num, pos)
        self.edit_rate = torch.mean(edit_locs, axis=1)

    def sample_words(self, probs, pos, sentences):
        chosen_words = torch.multinomial(torch.exp(probs), num_samples=1).squeeze(dim=-1)
        new_sentences = sentences.clone()
        new_sentences[:, pos] = chosen_words
        edit_loc = new_sentences[:, pos]!=sentences[:, pos]
        return new_sentences, edit_loc

    def mask_prob(self, position, sentences):
        masked_sentences = sentences.clone()
        masked_sentences[:, position] = self.mask_id
        logits = self.model(masked_sentences)[0]
        return F.log_softmax(logits[:, position] / self.temp, dim = -1)

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
    def __init__(self, sentences, temp, model, mask_id, num_masks):
        self.sentences = sentences
        self.temp = temp
        self.model = model
        self.mask_id = mask_id
        self.num_masks = num_masks
        self.edit_rate = torch.zeros(self.sentences.shape[0]).to(args.device)

    @torch.no_grad()
    def step(self, iter_num):
        seq_len = self.sentences.shape[1]
        # exclude first/last tokens (CLS/SEP) from positions
        mask_pos_list = (torch.randperm(seq_len-2)[:self.num_masks]+1).to(args.device)
        probs = self.mask_prob(mask_pos_list,self.sentences)
        self.sentences, edit_rate = self.sample_words(probs, mask_pos_list, self.sentences)
        self.edit_rate = edit_rate.mean(axis=1)

    def sample_words(self, probs, pos, sentences):
        old_score = self.get_total_score(sentences)
        old_words = sentences[:,pos]
        #Propose a set of words
        new_words = torch.tensor([list(torch.multinomial(prob, num_samples=1).squeeze(dim=-1)) for prob in torch.exp(probs)]).to(args.device)
        new_sentences = sentences.clone()
        new_sentences[:,pos] = new_words
        new_score = self.get_total_score(new_sentences)
        fwd_prob = torch.tensor([prob.index_select(dim=-1,index=word_list).diagonal().sum().item()\
                                for word_list,prob in zip(new_words,probs)]).to(args.device)
        bck_prob = torch.tensor([prob.index_select(dim=-1,index=word_list).diagonal().sum().item()\
                                for word_list,prob in zip(old_words,probs)]).to(args.device)
        alpha = torch.exp(new_score - old_score + bck_prob - fwd_prob)
        alpha[alpha>1] = 1
        accept = torch.rand(sentences.shape[0]).to(args.device)<alpha
        chosen_words = old_words.clone()
        chosen_words[accept,:] = new_words[accept,:]
        chosen_sentences = sentences.clone()
        chosen_sentences[:,pos] = chosen_words
        edit_rate = chosen_sentences[:,pos]!=sentences[:,pos]
        return chosen_sentences, edit_rate.float()

    def mask_prob(self, position, sentences):
        masked_sentences = sentences.clone()
        masked_sentences[:, position] = self.mask_id
        logits = self.model(masked_sentences)[0]
        return F.log_softmax(logits[:, position] / self.temp, dim = -1)

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
    if args.sampling_method=='gibbs_mixture':
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
        +f'bert_new_{args.sampling_method}_{args.epsilon}_{args.sentence_id}_{args.temp}.csv'
    else:
        f = f'BertData/{args.num_tokens}TokenSents/textfile/{args.model_name}/{args.batch_size}_{args.chain_len}/'\
        +f'bert_new_{args.sampling_method}_{args.sentence_id}_{args.temp}.csv'
    writer = Writer(args, f)

    for i, input_sentence in enumerate(input_sentences):
        print(f'Beginning batch {i}')
        start = time.time()

        # set up the input
        words = input_sentence.capitalize()
        tokenized_sentence = tokenizer(words, return_tensors="pt")
        assert len(tokenized_sentence["input_ids"][0]) == args.num_tokens, 'number of tokens in the initial sentence does not match the num_tokens argument'
        init_input = (tokenized_sentence["input_ids"][0]
                      .to(args.device)
                      .expand((args.batch_size, -1)))#, init_sentence.shape[0])))
        # reset writer
        writer.reset(i, words)

        # set up the sampler
        if 'gibbs' in args.sampling_method:
            sampler = UniformGibbs(init_input, args.temp, model, mask_id)
        elif 'mh' in args.sampling_method:
            sampler = MultiSiteMH(init_input, args.temp, model, mask_id, args.num_masks)

        switch = 0
        num_switches = 0
        for iter_num in range(args.chain_len):
            # write out substeps (gibbs) / all steps (mh) for iter_num<100
            if iter_num<100:
                if 'gibbs' in args.sampling_method:
                    seq_len = sampler.sentences.shape[1]
                    # exclude first/last tokens (CLS/SEP) from positions
                    rand_list = torch.randperm(seq_len-2).to(args.device)+1
                    for pos_id,pos in enumerate(rand_list):
                        writer.write(iter_num, sampler.sentences,
                                     sampler.get_total_score(sampler.sentences).cpu().detach().numpy(),
                                     sampler.edit_rate,pos_id,0)
                        sampler.sentences, sampler.edit_rate = sampler.substep(iter_num, pos)
                elif 'mh' in args.sampling_method:
                    writer.write(iter_num, sampler.sentences,
                                 sampler.get_total_score(sampler.sentences).cpu().detach().numpy(),
                                 sampler.edit_rate,sampler.sentences.shape[1]-3,0)
                    sampler.step(iter_num)
            else:
                if iter_num % args.sent_sample == 0:
                    writer.write(iter_num, sampler.sentences,
                                 sampler.get_total_score(sampler.sentences).cpu().detach().numpy(),
                                 sampler.edit_rate,sampler.sentences.shape[1]-3,switch)
                    switch = 0
                sampler.step(iter_num)
                if args.sampling_method=='gibbs_mixture' and iter_num>=1000 and torch.rand(1)<args.epsilon:
                    switch = 1
                    num_switches += 1

                    # Choose random wikipedia samples
                    sampled_ids = torch.randperm(wiki_samples.shape[0])[:args.batch_size].to(args.device)
                    new_init_input = wiki_samples[sampled_ids].to(args.device)

                    #sampler = UniformGibbs(init_input, args.temp, model, mask_id)
                    sampler = UniformGibbs(new_init_input, args.temp, model, mask_id)
                    for _ in range(100):
                        sampler.step(iter_num)
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
    args = parser.parse_args()

    if 'mh' in args.sampling_method:
        args.num_masks = int(args.sampling_method.split('_')[1])

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
    mask_id = tokenizer.encode("[MASK]")[1:-1][0]

    # launch chains
    run_chains(args)
