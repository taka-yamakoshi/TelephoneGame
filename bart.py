import torch
import numpy as np
import pickle
from multiprocessing import Pool
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import sys
sys.path.append('..')
args = sys.argv

# see ``examples/summarization/bart/evaluate_cnn.py`` for a longer example
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model.eval()
def Denoising(sentence):
    inputs = tokenizer.batch_encode_plus([sentence], return_tensors='pt')['input_ids']
    output_ids = model.generate(inputs,num_beams=10,do_sample=True,no_repeat_ngram_size=2)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
sentence = " Mary took a man who was wearing a hat the blanket."
prev_sentence = " "
while prev_sentence != sentence:
    prev_sentence = sentence
    sentence = Denoising(prev_sentence)
    print(sentence)
