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
import math


class JointProbTest() :
    def __init__(self, sentences, temp, model, mask_id, num_masks) :
        self.sentences = sentences
        self.temp = temp
        self.model = model
        self.mask_id = mask_id
        self.num_masks = num_masks
    
    @torch.no_grad()
    def get_joint_probs_all(self):
        seq_len = self.sentences.shape[1]
        # exclude first/last tokens (CLS/SEP) from positions
        mask_pos_list = (torch.randperm(seq_len - 2)[:self.num_masks]+1).to(args.device)
        joint_probs = torch.ones((self.sentences.shape[0],8)).to(args.device)
        joint_probs[:,0] = self.get_joint_probability_rand_field(mask_pos_list,self.sentences)
        joint_probs[:,1] = self.get_joint_probability(mask_pos_list,self.sentences,m=10)
        joint_probs[:,2] = self.get_joint_probability(mask_pos_list,self.sentences,m=20)
        joint_probs[:,3] = self.get_joint_probability(mask_pos_list,self.sentences,m=50)
        joint_probs[:,4] = self.get_joint_probability(mask_pos_list,self.sentences,m=100)
        joint_probs[:,5] = self.get_joint_probability(mask_pos_list,self.sentences,m=1000)
        joint_probs[:,6] = self.get_joint_probability_real(mask_pos_list,self.sentences)
        joint_probs[:,7] = self.get_joint_probability_real_new(mask_pos_list,self.sentences)
        return joint_probs,mask_pos_list
    
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
                                                          self.get_joint_probability(sampling_sites[sampling_sites!=site],sampled_sentences,m))
                joint_probs[:,iter_id] = torch.div(conditional,inv_prob_array.mean(dim=1))
            return torch.log(joint_probs.mean(dim=1))
        
    def get_joint_probability_real(self,sampling_sites,sentences):
        if len(sampling_sites) == 1:
            probs = self.mask_prob(sampling_sites.item(),sentences)
            conditional = torch.tensor([prob[word].item() for prob,word in zip(probs,sentences[:,sampling_sites.item()])]).to(args.device)
            return conditional
        else:
            random_id = 0
            site = sampling_sites[random_id]
            #Calculate the conditional probability
            probs = self.mask_prob(site,sentences)
            conditional = torch.tensor([prob[word].item() for prob,word in zip(probs,sentences[:,site])]).to(args.device)
            inv_prob_array = torch.zeros(probs.shape).to(args.device)
            assert probs.shape[0]==sentences.shape[0]
            for batch_id in range(sentences.shape[0]):
                sampled_sentences = sentences[batch_id].clone().to(args.device).expand((probs.shape[1],-1)).clone()
                sampled_sentences[:,site] = torch.arange(probs.shape[1]).to(args.device)
                #We have finite RAM so we need to batchify again 
                assert inv_prob_array.shape[1]==sampled_sentences.shape[0]
                if args.num_tokens <= 10:
                    new_batch_size = 5000
                elif args.num_tokens <= 20:
                    new_batch_size = 2500
                elif args.num_tokens <= 30:
                    new_batch_size = 1500
                new_batch_num = math.ceil(sampled_sentences.shape[0]/new_batch_size)
                for j in range(new_batch_num):
                    inv_prob_array[batch_id,new_batch_size*j:new_batch_size*(j+1)] = \
                    torch.div(torch.ones(sampled_sentences[new_batch_size*j:new_batch_size*(j+1)].shape[0]).to(args.device),\
                              self.get_joint_probability_real(sampling_sites[sampling_sites!=site],\
                                                              sampled_sentences[new_batch_size*j:new_batch_size*(j+1)]))
            return torch.log(torch.div(conditional,inv_prob_array.mean(dim=1)))
        
    def get_joint_probability_real_new(self,sampling_sites,sentences):
        if len(sampling_sites) == 1:
            probs = self.mask_prob(sampling_sites.item(),sentences)
            conditional = torch.tensor([prob[word].item() for prob,word in zip(probs,sentences[:,sampling_sites.item()])]).to(args.device)
            return conditional
        else:
            #Random list for which site to factor out
            random_list = torch.randperm(len(sampling_sites)).to(args.device)
            joint_probs = torch.zeros(sentences.shape[0],len(sampling_sites)).to(args.device)
            for iter_id,random_id in enumerate(random_list):
                site = sampling_sites[random_id]
                #Calculate the conditional probability
                probs = self.mask_prob(site,sentences)
                conditional = torch.tensor([prob[word].item() for prob,word in zip(probs,sentences[:,site])]).to(args.device)
                inv_prob_array = torch.zeros(probs.shape).to(args.device)
                assert probs.shape[0]==sentences.shape[0]
                for batch_id in range(sentences.shape[0]):
                    sampled_sentences = sentences[batch_id].clone().to(args.device).expand((probs.shape[1],-1)).clone()
                    sampled_sentences[:,site] = torch.arange(probs.shape[1]).to(args.device)
                    #We have finite RAM so we need to batchify again 
                    assert inv_prob_array.shape[1]==sampled_sentences.shape[0]
                    if args.num_tokens <= 10:
                        new_batch_size = 5000
                    elif args.num_tokens <= 20:
                        new_batch_size = 2500
                    elif args.num_tokens <= 30:
                        new_batch_size = 1500
                    new_batch_num = math.ceil(sampled_sentences.shape[0]/new_batch_size)
                    for j in range(new_batch_num):
                        inv_prob_array[batch_id,new_batch_size*j:new_batch_size*(j+1)] = \
                        torch.div(torch.ones(sampled_sentences[new_batch_size*j:new_batch_size*(j+1)].shape[0]).to(args.device),\
                                  self.get_joint_probability_real(sampling_sites[sampling_sites!=site],\
                                                                  sampled_sentences[new_batch_size*j:new_batch_size*(j+1)]))
                joint_probs[:,iter_id] = torch.div(conditional,inv_prob_array.mean(dim=1))
            return torch.log(joint_probs.mean(dim=1))
        
    def get_joint_probability_rand_field(self,sampling_sites,sentences):
        probs = self.mask_prob(sampling_sites,sentences)
        conditionals = torch.tensor([[probs[batch_id,site_id,word] for batch_id,word in enumerate(sentences[:,site])]\
                                     for site_id,site in enumerate(sampling_sites)]).to(args.device)
        return torch.log(conditionals).sum(dim=0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentence_id', type = str, required = True)
    parser.add_argument('--core_id', type = str, required = True)
    parser.add_argument('--num_tokens',type=int, required = True,
                        help='number of tokens including special tokens')
    parser.add_argument('--model_name', type = str, default = 'bert-base-uncased')
    parser.add_argument('--temp', type = float, default = 1, 
                        help='softmax temperature')
    parser.add_argument('--num_masks', type=int, required = True,
                        help='number of positions to sample at one time')
    parser.add_argument('--batch_size',type=int, default = 5)
    args = parser.parse_args()

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

    with open(f'WikiData/TokenSents/{args.num_tokens}TokenSents/textfile/{args.sentence_id}.txt','r') as f:
        loaded_sentences = f.read().split('\n')[:-1]
    #batch_num = math.ceil(len(loaded_sentences)/args.batch_size)
    #batched_sentences = [loaded_sentences[args.batch_size*i:args.batch_size*(i+1)] for i in range(batch_num)]
    #for sentences in batched_sentences:
    sentences = loaded_sentences[:args.batch_size]
    input_sentences = tokenizer(sentences,return_tensors='pt')["input_ids"].to(args.device)
    assert input_sentences.shape[1] == args.num_tokens
    
    test_model = JointProbTest(input_sentences, args.temp, model, mask_id, args.num_masks)
    joint_probs,mask_pos_list = test_model.get_joint_probs_all()
    output_dict = {}
    output_dict['sentences'] = sentences
    output_dict['joint_probs'] = joint_probs.to('cpu')
    output_dict['sampling_sites'] = mask_pos_list.to('cpu')
    
    with open(f'joint_probs_{args.num_tokens}_{args.sentence_id}.pkl','wb') as f:
        pickle.dump(output_dict,f)

