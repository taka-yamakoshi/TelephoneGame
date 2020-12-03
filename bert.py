# python bert.py --sentence_id input_1 --core_id 1
import torch
import numpy as np
import pickle
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import torch.nn.functional as F
import csv
import time
import sys
import os
import argparse
import spacy

# Load the model
nlp = spacy.load('en_core_web_lg')
tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
model = BertForMaskedLM.from_pretrained('bert-large-cased')

nlp.tokenizer.add_special_case("[UNK]",[{"ORTH": "[UNK]"}])
mask_id = tokenizer.encode("[MASK]")[1:-1][0]
sys.path.append('..')

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
            csv.writer(writeFile).writerows(header)

    def reset(self, sentence_num, init_sent) :
        self.init_sent = init_sent
        self.sentence_num = sentence_num

    def write(self, iter_num, sentences, probs):
        with open(self.save_file, "a") as writeFile:
            csvwriter = csv.writer(writeFile)
            decoded_sentences = [str(tokenizer.decode(sentence)) 
                                 for sentence in sentences]
            print(iter_num)
            print(decoded_sentences)
            for row_id, sentence in enumerate(decoded_sentences) :
                csvwriter.writerow([
                    self.sentence_num, row_id, iter_num, 
                    self.init_sent, sentence, probs[row_id]
                ])

class UniformGibbs() :
    def __init__(self, sentences, temp, fix_length) :
        self.sentences = sentences 
        self.fix_length = fix_length
        self.temp = temp

    @torch.no_grad()
    def step(self, iter_num) :
        seq_len = self.sentences.shape[1]

        # exclude first/last tokens (CLS/SEP) from positions
        rand_list = np.random.permutation(seq_len - 2) + 1
        for pos_id,pos in enumerate(rand_list):
            probs = self.mask_prob(pos)
            self.sentences = self.sample_words(probs, pos, self.sentences)

            # keep sampling until they're the correct length according to spacy
            attempts = 0
            while not all([len(nlp(tokenizer.decode(sentence[1:-1]))) == 11 
                           for sentence in self.sentences]) and self.fix_length:

                # resample only indices that are bad
                bad_i = torch.ByteTensor([
                    len(nlp(tokenizer.decode(sentence[1:-1]))) != 11 
                    for sentence in self.sentences
                ])
                self.sentences[bad_i, :] = self.sample_words(probs[bad_i,:], pos, 
                                                             self.sentences[bad_i, :])
                # sometimes we get stuck with values where there are no valid choices.
                # in this case, just move on.
                attempts += 1
                if attempts > 10 :
                    break

    def sample_words(self, probs, pos, sentences):
        chosen_words = torch.multinomial(probs, 1).squeeze(-1)
        new_sentences = sentences.clone()
        new_sentences[:,pos] = chosen_words
        return new_sentences

    def mask_prob (self, position) :
        """
        Predict probability of words at mask position
        """
        masked_sentences = self.sentences.clone()
        masked_sentences[:, position] = mask_id
        outputs = model(masked_sentences)
        return F.softmax(outputs[0][:, position] / self.temp, dim = -1)

    def get_total_likelihood(self) :
        sent_probs = np.zeros(self.sentences.shape)

        # Why cut off first and last?
        for j in range(1, self.sentences.shape[1] - 1) :
            probs = torch.log(self.mask_prob(j))
            for i in range(self.sentences.shape[0]) :
                # Look up probability of the actual word at this position
                sent_probs[i, j] = probs[i, self.sentences[i, j]].item() 

        # Sum log probs for each sentence in batch
        return np.sum(sent_probs, axis=1)

def run_chains(args) :
    # Load sentences
    with open(f'{args.sentence_id}.txt','r') as f:
        input_sentences = f.read().split('\n')[:-1]
        batch_num = len(input_sentences)

    # Run the sampling
    f = f'textfile/bert_{args.sampling_method}_{args.sentence_id}_{args.temp}.csv'
    writer = Writer(args, f)
    for i, input_sentence in enumerate(input_sentences):
        print(f'Beginning batch {i}')
        time1 = time.time()
        words = input_sentence.capitalize() + "."
        tokenized_sentence = tokenizer(words, return_tensors="pt")
        init_input = (tokenized_sentence["input_ids"][0]
                      .to(args.device)
                      .expand((args.batch_size, -1)))#, init_sentence.shape[0])))
        # reset writer 
        writer.reset(i, words)
        sampler = UniformGibbs(init_input, args.temp, args.fix_length)
        for iter_num in range(args.chain_len):
            print(f'Beginning iteration {iter_num}')
            sampler.step(iter_num)
            
            # Write out sentences
            if iter_num % args.sent_sample == 0:
                writer.write(iter_num, sampler.sentences, 
                             sampler.get_total_likelihood())
        time2 = time.time()
        print(f'Time it took for {i}th batch: {time2-time1}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentence_id', type = str, required = True)
    parser.add_argument('--core_id', type = str, required = True)
    parser.add_argument('--batch_size', type=int, default=20, 
                        help='number of sentences to maintain')
    parser.add_argument('--chain_len', type=int, default = 10000, 
                        help='number of samples')
    parser.add_argument('--temp', type = float, default = 1, 
                        help='softmax temperature')
    parser.add_argument('--sent_sample', type=int, default = 5, 
                        help='frequency of recording sentences')
    parser.add_argument('--sampling_method', type=str, default = 'gibbs', 
                        choices=['gibbs'],
                        help='kind of sampling to do; options include "gibbs"')
    parser.add_argument('--fix_length', type=bool, default = False, 
                        help='if True, resample to avoid changing length')

    args = parser.parse_args()

    print('running with args', args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.core_id
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    model.to(args.device)
    model.eval()

    # launch chains
    run_chains(args)
