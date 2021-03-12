# python bert.py --sentence_id input_1 --core_id 1
import torch
import numpy as np
import pickle
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from ExtractFixedLenSents import TokenizerSetUp
import torch.nn.functional as F
import csv
import time
import sys
import os
import argparse
import spacy


class Writer():
    def __init__(self, args, save_file) :
        self.batch_size = args.batch_size
        self.chain_len = args.chain_len
        self.save_file = save_file
        self.init_output_csv()

    def init_output_csv(self, header=None):
        header = ['sentence_num','chain_num','iter_num',
                  'initial_sentence', 'sentence', 'prob', 'edit_rate']
        with open(self.save_file, "w") as writeFile:
            csv.writer(writeFile).writerow(header)

    def reset(self, sentence_num, init_sent) :
        self.init_sent = init_sent
        self.sentence_num = sentence_num

    def write(self, iter_num, sentences, probs, edit_rate):
        with open(self.save_file, "a") as writeFile:
            csvwriter = csv.writer(writeFile)
            decoded_sentences = [str(tokenizer.decode(sentence)) 
                                 for sentence in sentences]
            if iter_num%1000==0:
                print(iter_num)
                print(decoded_sentences)
                print(probs)
                print(edit_rate)
            for row_id, sentence in enumerate(decoded_sentences) :
                csvwriter.writerow([
                    self.sentence_num, row_id, iter_num, 
                    self.init_sent, sentence, probs[row_id], edit_rate[row_id].item()
                ])

class UniformGibbs() :
    def __init__(self, sentences, temp, fix_length, model, mask_id) :
        self.sentences = sentences 
        self.fix_length = fix_length
        self.temp = temp
        self.model = model
        self.mask_id = mask_id

    @torch.no_grad()
    def step(self, iter_num) :
        seq_len = self.sentences.shape[1]

        # exclude first/last tokens (CLS/SEP) from positions
        rand_list = np.random.permutation(seq_len - 2) + 1
        edit_locs = torch.zeros((self.sentences.shape[0],len(rand_list)))
        for pos_id,pos in enumerate(rand_list):
            probs = self.mask_prob(pos)
            self.sentences, edit_loc = self.sample_words(probs, pos, self.sentences)

            # keep sampling until they're the correct length according to spacy
            attempts = 0
            if self.fix_length:
                while not all([len(nlp(tokenizer.decode(sentence[1:-1]))) == 11
                               for sentence in self.sentences]):

                    # resample only indices that are bad
                    bad_i = torch.ByteTensor([
                        len(nlp(tokenizer.decode(sentence[1:-1]))) != 11
                        for sentence in self.sentences
                    ])
                    self.sentences[bad_i, :], edit_loc = self.sample_words(probs[bad_i,:], pos,
                                                                 self.sentences[bad_i, :])
                    # sometimes we get stuck with values where there are no valid choices.
                    # in this case, just move on.
                    attempts += 1
                    if attempts > 10 :
                        break
            edit_locs[:,pos_id] = edit_loc
        self.edit_rate = torch.mean(edit_locs,axis=1)

    def sample_words(self, probs, pos, sentences):
        chosen_words = torch.multinomial(probs, 1).squeeze(-1)
        new_sentences = sentences.clone()
        new_sentences[:,pos] = chosen_words
        edit_loc = new_sentences[:,pos]!=sentences[:,pos]
        return new_sentences, edit_loc

    def mask_prob(self, position):
        """
        Predict probability of words at mask position
        """
        masked_sentences = self.sentences.clone()
        masked_sentences[:, position] = self.mask_id
        outputs = self.model(masked_sentences)
        return F.softmax(outputs[0][:, position] / self.temp, dim = -1)

    def get_total_likelihood(self):
        sent_probs = torch.zeros(self.sentences.shape).to(args.device)

        # Why cut off first and last?
        for j in range(1, self.sentences.shape[1] - 1) :
            probs = torch.log(self.mask_prob(j))
            for i in range(self.sentences.shape[0]) :
                # Look up probability of the actual word at this position
                sent_probs[i, j] = probs[i, self.sentences[i, j]]

        # Sum log probs for each sentence in batch
        return torch.sum(sent_probs, axis=1)

class MultiSiteMH() :
    def __init__(self, sentences, temp, fix_length, model, mask_id, num_masks) :
        self.sentences = sentences
        self.fix_length = fix_length
        self.temp = temp
        self.model = model
        self.mask_id = mask_id
        self.num_masks = num_masks
    
    @torch.no_grad()
    def step(self, iter_num) :
        seq_len = self.sentences.shape[1]
        
        # exclude first/last tokens (CLS/SEP) from positions
        mask_pos_list = (torch.randperm(seq_len - 2)[:self.num_masks]+1).to(args.device)
        probs = self.mask_prob(mask_pos_list,self.sentences)
        self.sentences, edit_rate = self.sample_words(probs, mask_pos_list, self.sentences)
        self.edit_rate = edit_rate.mean(axis=1)
    
    def sample_words(self, probs, pos, sentences):
        old_joint_prob = self.get_joint_probability(pos,sentences)
        old_words = sentences[:,pos]
        #Propose a set of words
        new_words = torch.tensor([list(torch.multinomial(prob, 1).squeeze(-1)) for prob in probs]).to(args.device)
        new_sentences = sentences.clone()
        new_sentences[:,pos] = new_words
        new_joint_prob = self.get_joint_probability(pos,new_sentences)
        fwd_prob = torch.tensor([[torch.log(prob[word]).item() for word,prob in zip(word_list,prob_array)]\
                                     for word_list,prob_array in zip(new_words,probs)]).sum(axis=1).to(args.device)
        bck_prob = torch.tensor([[torch.log(prob[word]).item() for word,prob in zip(word_list,prob_array)]\
                                     for word_list,prob_array in zip(old_words,probs)]).sum(axis=1).to(args.device)
        alpha = torch.exp(new_joint_prob - old_joint_prob + bck_prob - fwd_prob)
        alpha[alpha>1] = 1
        accept = torch.rand(sentences.shape[0]).to(args.device)<alpha
        chosen_words = old_words.clone()
        chosen_words[accept,:] = new_words[accept,:]
        chosen_sentences = sentences.clone()
        chosen_sentences[:,pos] = chosen_words
        edit_rate = chosen_sentences[:,pos]!=sentences[:,pos]
        return chosen_sentences, edit_rate.float()
    
    def mask_prob(self, position, sentences):
        """
            Predict probability of words at mask position
            This is the same as UniformGibbs.mask_prob, except the input now includes sentences
            """
        masked_sentences = sentences.clone()
        masked_sentences[:, position] = self.mask_id
        outputs = self.model(masked_sentences)
        return F.softmax(outputs[0][:, position] / self.temp, dim = -1)
    
    def get_total_likelihood(self):
        """
            This is the same as UniformGibbs.get_total_likelihood
            """
        sent_probs = torch.zeros(self.sentences.shape).to(args.device)
        
        # Why cut off first and last?
        for j in range(1, self.sentences.shape[1] - 1) :
            probs = torch.log(self.mask_prob(j,self.sentences))
            for i in range(self.sentences.shape[0]) :
                # Look up probability of the actual word at this position
                sent_probs[i, j] = probs[i, self.sentences[i, j]]
    
        # Sum log probs for each sentence in batch
        return torch.sum(sent_probs, axis=1)
    
    def get_joint_probability(self,sampling_sites,sentences,m=10):
        if len(sampling_sites) == 1:
            probs = self.mask_prob(sampling_sites.item(),sentences)
            conditional = torch.tensor([prob[word].item() for prob,word in zip(probs,sentences[:,sampling_sites.item()])]).to(args.device)
            return conditional
        else:
            #Random list for which site to factor out
            random_list = torch.randperm(len(sampling_sites)).to(args.device)
            joint_probs = torch.zeros(sentences.shape[0],len(sampling_sites)).to(args.device)
            for iter_id, random_id in enumerate(random_list):
                #Pick a site to factor out
                site = sampling_sites[random_id]
                #Calculate the conditional probability
                probs = self.mask_prob(site,sentences)
                conditional = torch.tensor([prob[word].item() for prob,word in zip(probs,sentences[:,site])]).to(args.device)
                #Sample for Monte-Carlo
                sampled_words = torch.tensor([list(torch.multinomial(prob, m, replacement=True).squeeze(-1)) for prob in probs]).to(args.device)
                inv_prob_array = torch.zeros(sampled_words.shape).to(args.device)
                for sample_iter in range(m):
                    sampled_sentences = sentences.clone()
                    sampled_sentences[:,site] = sampled_words[:,sample_iter]
                    inv_prob_array[:,sample_iter] = torch.div(torch.ones(sentences.shape[0]).to(args.device),\
                                                          self.get_joint_probability(sampling_sites[sampling_sites!=site],sampled_sentences))
                joint_probs[:,iter_id] = torch.div(conditional,inv_prob_array.mean(dim=1))
            return torch.log(joint_probs.mean(dim=1))
            

def run_chains(args) :
    # Load sentences
    with open(f'initial_sentences/{args.num_tokens}Tokens/{args.sentence_id}.txt','r') as f:
        input_sentences = f.read().split('\n')[:-1]
        batch_num = len(input_sentences)

    # Run the sampling
    os.makedirs(f'BertData/{args.num_tokens}TokenSents/textfile/{args.model_name}/{args.batch_size}_{args.chain_len}/', exist_ok=True)
    f = f'BertData/{args.num_tokens}TokenSents/textfile/{args.model_name}/{args.batch_size}_{args.chain_len}/'\
        +f'bert_{args.sampling_method}_{args.num_masks}_{args.sentence_id}_{args.temp}.csv'
    writer = Writer(args, f)
    for i, input_sentence in enumerate(input_sentences):
        print(f'Beginning batch {i}')
        time1 = time.time()
        words = input_sentence.capitalize() + "."
        tokenized_sentence = tokenizer(words, return_tensors="pt")
        assert len(tokenized_sentence["input_ids"][0]) == args.num_tokens, 'number of tokens in the initial sentence does not match the num_tokens argument'
        init_input = (tokenized_sentence["input_ids"][0]
                      .to(args.device)
                      .expand((args.batch_size, -1)))#, init_sentence.shape[0])))
        # reset writer 
        writer.reset(i, words)
        if args.sampling_method == 'gibbs':
            sampler = UniformGibbs(init_input, args.temp, args.fix_length, model, mask_id)
        elif args.sampling_method == 'mh':
            sampler = MultiSiteMH(init_input, args.temp, args.fix_length, model, mask_id, args.num_masks)
        for iter_num in range(args.chain_len):
            #print(f'Beginning iteration {iter_num}')
            sampler.step(iter_num)
            
            # Write out sentences
            if iter_num % args.sent_sample == 0:
                writer.write(iter_num, sampler.sentences,
                             sampler.get_total_likelihood().cpu().detach().numpy(),sampler.edit_rate)
        time2 = time.time()
        print(f'Time it took for {i}th batch: {time2-time1}')

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
    parser.add_argument('--sampling_method', type=str, default = 'gibbs', 
                        choices=['gibbs','mh'],
                        help='kind of sampling to do; options include "gibbs","mh"')
    parser.add_argument('--fix_length', type=bool, default = False, 
                        help='if True, resample to avoid changing length')
    parser.add_argument('--num_masks', type=int, default = 1,
                        help='number of positions to sample at one time')
    args = parser.parse_args()

    print('running with args', args)

    if args.fix_length:
        nlp = TokenizerSetUp()
    
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

