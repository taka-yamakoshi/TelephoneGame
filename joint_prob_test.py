# python joint_prob_test.py --sentence_id input_0 --batch_size 1 --num_tokens 12 --num_masks 2 --core_id 1
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

# set random seed
torch.manual_seed(0)

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
        out = {}        

        # exclude first/last tokens (CLS/SEP) from positions
        mask_pos_list = (torch.randperm(seq_len - 2)[:self.num_masks]+1).to(args.device)

        start = time.time()
        out['real'] = self.get_joint_probability_bidirectional(mask_pos_list,self.sentences).to('cpu')
        print(out)
        print(f"Time it took for real new: {time.time()-start}")

        # sample_size_list = [10,100,1000,10000]
        # for sample_size in sample_size_list:
        #     start = time.time()
        #     out[f'MC_{sample_size}'] = torch.log(self.get_joint_probability(mask_pos_list,self.sentences,m=sample_size)).to('cpu')
        #     print(f"Time it took for MC with {sample_size} {time.time()-start}")
        
        # start = time.time()
        # out['rand_field'] = torch.log(self.get_joint_probability_rand_field(mask_pos_list,self.sentences)).to('cpu')
        # print(f"Time it took for rand_field: {time.time()-start}")
        
        # start = time.time()
        # out['mask'] = torch.log(self.get_joint_probability_mask(mask_pos_list,self.sentences)).to('cpu')
        # print(f"Time it took for mask: {time.time()-start}")
                
        # start = time.time()
        # out['real_new'] = torch.log(self.get_joint_probability_real_new(mask_pos_list,self.sentences)).to('cpu')
        # print(f"Time it took for real_new: {time.time()-start}")
        return out,mask_pos_list.to('cpu')

    def get_joint_probability_bidirectional(self,sampling_sites,sentences):
        '''
            Calculate the "real" joint probability by taking the actual sum over the entire vocabulary
            Averaging multiple ways of factoring out
            Not practical for more than two sites
        '''
        # When sampling at more than one site, express joint probability in terms of joint probability with one less sites
        # Random list for which site to factor out
        random_list = torch.randperm(len(sampling_sites)).to(args.device)
        joint_probs = torch.zeros(sentences.shape[0],len(sampling_sites)).to(args.device)
        
        # Loop through different ways to factor out sites
        for iter_id,random_id in enumerate(random_list):
            # Pick a site to factor out
            site = sampling_sites[random_id]
            new_sampling_sites = sampling_sites.clone()
            new_sampling_sites[0] = site
            new_sampling_sites[1:] = sampling_sites[sampling_sites != site]
            joint_probs[:,iter_id] = self.get_joint_logprobability_real(new_sampling_sites,sentences)
        print('two directions', joint_probs)
        return joint_probs.mean(dim=1)

    def get_joint_logprobability_real(self,sampling_sites,sentences):
        '''
        Calculate the "real" joint probability by taking the actual sum over the entire vocabulary
        Not practical for more than two sites
        '''
        # When sampling at a single site, return conditional
        if len(sampling_sites) == 1:
            logprobs = self.mask_prob(sampling_sites.item(), sentences)
            true_word = sentences[:,sampling_sites.item()]
            return torch.diagonal(torch.index_select(logprobs, dim=1, index=true_word))
        
        # When sampling at more than one site, use joint probability with one less site
        else:
            # Always factor out the first site
            random_id = 0
            site = sampling_sites[random_id]

            # Calculate the conditional probability of the word at this site
            conditional = self.get_joint_logprobability_real(site.unsqueeze(0), sentences)
            logprobs = self.mask_prob(site,sentences)
            vocab_size = logprobs.shape[1]
            inv_prob_array = torch.zeros(logprobs.shape).to(args.device)
            assert logprobs.shape[0]==sentences.shape[0]
            
            # Loop through sentences in a batch
            for i in range(sentences.shape[0]):
                # Create every possible sentences, spanning the entire vocabulary
                sampled_sentences = sentences[i].clone().to(args.device).expand((vocab_size,-1)).clone()
                sampled_sentences[:,site] = torch.arange(vocab_size).to(args.device)
                
                # We have finite RAM so we need to batchify again
                assert inv_prob_array.shape[1]==sampled_sentences.shape[0]
                if args.num_tokens <= 10:
                    new_batch_size = 2000
                elif args.num_tokens <= 20:
                    new_batch_size = 1000
                elif args.num_tokens <= 30:
                    new_batch_size = 500
                new_batch_num = math.ceil(sampled_sentences.shape[0]/new_batch_size)
                
                # Go through new batches
                for j in range(new_batch_num):
                    # Note we are using "logprobs" in the numerator
                    other_sites = sampling_sites[sampling_sites!=site]
                    batch = sampled_sentences[new_batch_size*j:new_batch_size*(j+1)]
                    term1 = logprobs[i,new_batch_size*j:new_batch_size*(j+1)]
                    term2 = self.get_joint_logprobability_real(other_sites, batch)
                    inv_prob_array[i,new_batch_size*j:new_batch_size*(j+1)] = term1 - term2

            # Note this is "sum" and not "mean"
            print('conditional', conditional)
            print('logsumexp', torch.logsumexp(inv_prob_array, 1))
            return conditional - torch.logsumexp(inv_prob_array, 1)
    
    def mask_prob(self, position, sentences):
        """Predict probability of words at mask position.

        This is the same as UniformGibbs.mask_prob, but taking sentences as input
        """
        masked_sentences = sentences.clone()
        masked_sentences[:, position] = self.mask_id
        outputs = self.model(masked_sentences)[0]
        return F.log_softmax(outputs[:, position] / self.temp, dim = -1)
    
    def get_total_likelihood(self):
        """
            This is the same as UniformGibbs.get_total_likelihood
            """
        sent_probs = torch.zeros(self.sentences.shape).to(args.device)
        
        # Why cut off first and last? -- We are not talking into accout how likely we see special tokens at the beggining and the end
        for j in range(1, self.sentences.shape[1] - 1) :
            probs = self.mask_prob(j,self.sentences)
            for i in range(self.sentences.shape[0]) :
                # Look up probability of the actual word at this position
                sent_probs[i, j] = probs[i, self.sentences[i, j]]
    
        # Sum log probs for each sentence in batch
        return torch.sum(sent_probs, axis=1)

    def get_joint_probability_old(self,sampling_sites,sentences,m=10):
        '''
            Calculate joint probability of a batch of sentences using Monte-Carlo approximation
            '''
        # When sampling at a single site, return conditional
        if len(sampling_sites) == 1:
            logprobs = self.mask_prob(sampling_sites.item(),sentences)
            conditional = torch.tensor([logprob[word].item() for logprob,word in zip(logprobs,sentences[:,sampling_sites.item()])]).to(args.device)
            return conditional
        
        # When sampling at more than one site, express joint probability in terms of joint probability with one less sites
        else:
            # Random list for which site to factor out
            random_list = torch.randperm(len(sampling_sites)).to(args.device)
            joint_probs = torch.zeros(sentences.shape[0],len(sampling_sites)).to(args.device)
            
            # Loop through different ways to factor out sites
            for iter_id, random_id in enumerate(random_list):
                # Pick a site to factor out
                site = sampling_sites[random_id]
                
                # Calculate the conditional probability
                probs = self.mask_prob(site,sentences)
                conditional = torch.tensor([prob[word].item() for prob,word in zip(probs,sentences[:,site])]).to(args.device)
                
                # Sample for Monte-Carlo
                sampled_words = torch.tensor([list(torch.multinomial(prob, m, replacement=True).squeeze(-1))\
                                              for prob in probs]).to(args.device)
                inv_prob_array = torch.zeros(sampled_words.shape).to(args.device)
                
                # Loop through Monte-Carlo samples
                for sample_iter in range(m):
                    sampled_sentences = sentences.clone()
                    sampled_sentences[:,site] = sampled_words[:,sample_iter]
                    
                    # Express the probability in terms of joint probability with one less sites
                    inv_prob_array[:,sample_iter] = torch.exp(-self.get_joint_probability(sampling_sites[sampling_sites!=site],sampled_sentences,m))
                
                assert torch.all(inv_prob_array!=0)
                # Approximate expectation by taking the average over Monte-Carlo samples
                joint_probs[:,iter_id] = torch.exp(conditional-torch.log(inv_prob_array.mean(dim=1)))
            # Take the average over different ways to factor out sites
            return joint_probs.mean(dim=1)
    
    def get_joint_probability(self,sampling_sites,sentences,m=10):
        '''
            Calculate joint probability of a batch of sentences using Monte-Carlo approximation
            '''
        # When sampling at a single site, return conditional
        if len(sampling_sites) == 1:
            logprobs = self.mask_prob(sampling_sites.item(),sentences)
            conditional = torch.tensor([prob[word].item() for prob,word in zip(logprobs,sentences[:,sampling_sites.item()])]).to(args.device)
            return conditional
        
        # When sampling at more than one site, express joint probability in terms of joint probability with one less sites
        else:
            # Random list for which site to factor out
            random_list = torch.randperm(len(sampling_sites)).to(args.device)
            joint_probs = torch.zeros(sentences.shape[0],len(sampling_sites)).to(args.device)
            
            # Loop through different ways to factor out sites
            for iter_id, random_id in enumerate(random_list):
                # Pick a site to factor out
                site = sampling_sites[random_id]
                
                # Calculate the conditional probability
                probs = self.mask_prob(site,sentences)
                conditional = torch.tensor([prob[word].item() for prob,word in zip(probs,sentences[:,site])]).to(args.device)
                
                # Sample for Monte-Carlo
                sampled_words = torch.tensor([list(torch.multinomial(prob, m, replacement=True).squeeze(-1))\
                                              for prob in probs]).to(args.device)
                inv_prob_array = torch.zeros(sampled_words.shape).to(args.device)
                
                for i in range(sentences.shape[0]):
                    # Remove the duplicates
                    new_sampled_words = torch.nonzero(sampled_words[i,:].bincount()).squeeze(dim=1)
                    sampled_sentences = sentences[i].clone().to(args.device).expand((len(new_sampled_words),-1)).clone()
                    sampled_sentences[:,site] = new_sampled_words
                    new_inv_prob_array = torch.exp(-self.get_joint_probability(sampling_sites[sampling_sites!=site],sampled_sentences,m))
                    inv_prob_array_emb = torch.zeros(probs.shape[1])
                    for word_id, word in enumerate(new_sampled_words):
                        inv_prob_array_emb[word] = new_inv_prob_array[word_id]
                    inv_prob_array[i,:] = inv_prob_array_emb[sampled_words[i,:]]
                assert torch.all(inv_prob_array!=0)
                # Approximate expectation by taking the average over Monte-Carlo samples
                joint_probs[:,iter_id] = torch.exp(conditional-torch.log(inv_prob_array.mean(dim=1)))
            # Take the average over different ways to factor out sites
            return joint_probs.mean(dim=1)
        
    def get_joint_probability_rand_field(self,sampling_sites,sentences):
        '''
            Calculate the joint probability in the Markov Random Field way
            '''
        probs = self.mask_prob(sampling_sites,sentences)
        conditionals = torch.tensor([[probs[batch_id,site_id,word] for batch_id,word in enumerate(sentences[:,site])]\
                                     for site_id,site in enumerate(sampling_sites)]).to(args.device)
        return torch.exp(conditionals.sum(dim=0))

    def get_joint_probability_mask(self,sampling_sites,sentences):
        '''
            Calculate the joing probability by approximating the marginal using masks
            '''
        if len(sampling_sites) == 1:
            probs = self.mask_prob(sampling_sites.item(),sentences)
            conditional = torch.tensor([prob[word].item() for prob,word in zip(probs,sentences[:,sampling_sites.item()])]).to(args.device)
            return conditional
        else:
            random_list = torch.randperm(len(sampling_sites)).to(args.device)
            joint_probs = torch.zeros(sentences.shape[0],len(sampling_sites)).to(args.device)
            # Loop through different ways to factor out sites
            for iter_id,random_id in enumerate(random_list):
                # Pick a site to factor out
                site = sampling_sites[random_id]
                probs = self.mask_prob(site,sentences)
                conditional = torch.tensor([prob[word].item() for prob,word in zip(probs,sentences[:,site])]).to(args.device)
                masked_sentences = sentences.clone()
                masked_sentences[:,site] = self.mask_id
                joint_probs[:,iter_id] = conditional*self.get_joint_probability_mask(sampling_sites[sampling_sites!=site],masked_sentences)
            return joint_probs.mean(dim=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentence_id', type = str, required = True)
    parser.add_argument('--core_id', type = str, required = True)
    parser.add_argument('--local_path', dest='local_path', action='store_true')
    parser.add_argument('--num_tokens',type=int, required = True,
                        help='number of tokens including special tokens')
    parser.add_argument('--model_name', type = str, default = 'bert-base-uncased')
    parser.add_argument('--temp', type = float, default = 1, 
                        help='softmax temperature')
    parser.add_argument('--num_masks', type=int, required = True,
                        help='number of positions to sample at one time')
    parser.add_argument('--batch_size',type=int, default = 5,
                        help='number of sentences to calculate joint probability')
    '''
    parser.add_argument('--metric',type=str, required=True,
                        choices = ['MC','real','real_new','rand_field','mask'],
                        help='the way to calculate the joint probability')
    parser.add_argument('--sample_size',type=int, default = 10,
                        help='the sample_size for MC')
    '''
    parser.set_defaults(local_path=False)
    args = parser.parse_args()

    print('running with args', args)
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertForMaskedLM.from_pretrained(args.model_name)
    if torch.cuda.is_available():
        args.device = torch.device("cuda", index=int(args.core_id))
    else:
        args.device = torch.device("cpu")
    model.to(args.device)
    model.eval()
    mask_id = tokenizer.encode("[MASK]")[1:-1][0]

    path = '' if args.local_path else 'WikiData/TokenSents/{args.num_tokens}TokenSents/textfile/'
    with open(f'{path}{args.sentence_id}.txt','r') as f:
        loaded_sentences = f.read().split('\n')[:-1]

    sentences = loaded_sentences[:args.batch_size]
    input_sentences = tokenizer(sentences,return_tensors='pt')["input_ids"].to(args.device)
    assert input_sentences.shape[1] == args.num_tokens

    test_model = JointProbTest(input_sentences, args.temp, model, mask_id, args.num_masks)
    joint_probs,mask_pos_list = test_model.get_joint_probs_all()

    output_dict = {}
    output_dict['sentences'] = sentences
    output_dict['joint_probs'] = joint_probs
    output_dict['sampling_sites'] = mask_pos_list

    with open(f'joint_probs_{args.num_tokens}_{args.sentence_id}.pkl','wb') as f:
        pickle.dump(output_dict,f)

