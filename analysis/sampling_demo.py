import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertForMaskedLM
import torch
import torch.nn.functional as F

@st.cache
def load_model(model_name):
    if model_name.startswith('bert'):
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForMaskedLM.from_pretrained(model_name)
    return tokenizer,model

def mask_prob(model,mask_id,sentences,position,temp=1):
    masked_sentences = sentences.clone()
    masked_sentences[:, position] = mask_id
    logits = model(masked_sentences)[0]
    return F.log_softmax(logits[:, position] / temp, dim = -1)

def sample_words(probs, pos, sentences):
    candidates = [[tokenizer.decode([candidate]),torch.exp(probs)[0,candidate].item()]
                  for candidate in torch.argsort(probs[0],descending=True)[:10]]
    df = pd.DataFrame(data=candidates,columns=['word','prob'])
    chosen_words = torch.multinomial(torch.exp(probs), num_samples=1).squeeze(dim=-1)
    new_sentences = sentences.clone()
    new_sentences[:, pos] = chosen_words
    return new_sentences, df

def initialize_sentence():
    del st.session_state['sentence']
    del st.session_state['pos']
    
max_width = 1500
padding_top = 2
padding_right = 5
padding_bottom = 0
padding_left = 5

st.markdown(
        f"""
<style>
    .appview-container .main .block-container{{
        max-width: {max_width}px;
        padding-top: {padding_top}rem;
        padding-right: {padding_right}rem;
        padding-left: {padding_left}rem;
        padding-bottom: {padding_bottom}rem;
    }}
</style>
""",
        unsafe_allow_html=True,
    )

hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """
st.markdown(hide_table_row_index, unsafe_allow_html=True)


#model_name = st.selectbox(label='Model',options=('bert-base-uncased','bert-large-cased'))
model_name = 'bert-base-uncased'
tokenizer,model = load_model(model_name)
mask_id = tokenizer.encode("[MASK]")[1:-1][0]

container = st.sidebar.container()
container.write('Instruction')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.text_input(label='You can also change the initial sentence here',value='About 170 campers attend the camps each week.',key='init_sent',on_change=initialize_sentence)
if 'sentence' not in st.session_state:
    st.session_state.sentence = tokenizer(st.session_state.init_sent,return_tensors='pt')['input_ids']
    st.session_state.decoded_sent = [tokenizer.decode([token]) for token in st.session_state.sentence[0]]

cols = st.columns(len(st.session_state.decoded_sent))
with cols[0]:
    st.write(st.session_state.decoded_sent[0])
for word_id,(word,col) in enumerate(zip(st.session_state.decoded_sent[1:-1],cols[1:-1])):
    with col:
        if st.button(word):
            st.session_state.pos = word_id + 1
with cols[-1]:
    st.write(st.session_state.decoded_sent[-1])

if 'pos' in st.session_state:
    probs = mask_prob(model,mask_id,st.session_state.sentence,st.session_state.pos)
    st.session_state.sentence,candidates_df = sample_words(probs,st.session_state.pos,st.session_state.sentence)
    with cols[st.session_state.pos]:
        st.table(candidates_df)
    st.session_state.decoded_sent = [tokenizer.decode([token]) for token in st.session_state.sentence[0]]
    del st.session_state['pos']
    container.write('Please press this to sample')
    cols = container.columns(3)
    with cols[1]:
        st.button('sample')
else:
    container.write('Please press a word you want to resample')