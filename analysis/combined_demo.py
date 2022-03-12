import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer,BertForMaskedLM

def load_sentence_model():
    sentence_model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    return sentence_model

@st.cache
def load_model(model_name):
    if model_name.startswith('bert'):
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForMaskedLM.from_pretrained(model_name)
        model.eval()
    return tokenizer,model

@st.cache
def load_data(sentence_num):
    df = pd.read_csv('movies/tsne_out.csv')
    df = df.loc[lambda d: (d['sentence_num']==sentence_num)&(d['iter_num']<1000)]
    return df

@st.cache
def mask_prob(model,mask_id,sentences,position,temp=1):
    masked_sentences = sentences.clone()
    masked_sentences[:, position] = mask_id
    with torch.no_grad():
        logits = model(masked_sentences)[0]
    return F.log_softmax(logits[:, position] / temp, dim = -1)

@st.cache
def sample_words(probs, pos, sentences):
    candidates = [[tokenizer.decode([candidate]),torch.exp(probs)[0,candidate].item()]
                  for candidate in torch.argsort(probs[0],descending=True)[:10]]
    df = pd.DataFrame(data=candidates,columns=['word','prob'])
    chosen_words = torch.multinomial(torch.exp(probs), num_samples=1).squeeze(dim=-1)
    new_sentences = sentences.clone()
    new_sentences[:, pos] = chosen_words
    return new_sentences, df

def run_chains(tokenizer,model,mask_id,input_text,num_steps):
    init_sent = tokenizer(input_text,return_tensors='pt')['input_ids']
    seq_len = init_sent.shape[1]
    sentence = init_sent.clone()
    data_list = []
    st.sidebar.write('Generating samples...')
    chain_progress = st.sidebar.progress(0)
    for step_id in range(num_steps):
        chain_progress.progress((step_id+1)/num_steps)
        pos = torch.randint(seq_len-2,size=(1,)).item()+1
        data_list.append([step_id,' '.join([tokenizer.decode([token]) for token in sentence[0]]),pos])
        probs = mask_prob(model,mask_id,sentence,pos)
        sentence,_ = sample_words(probs,pos,sentence)
    return pd.DataFrame(data=data_list,columns=['step','sentence','next_sample_loc'])

@st.cache(suppress_st_warning=True)
def run_tsne(chain):
    st.sidebar.write('Running t-SNE...')
    chain = chain.assign(cleaned_sentence=chain.sentence.str.replace(r'\[CLS\] ', '',regex=True).str.replace(r' \[SEP\]', '',regex=True))
    sentence_model = load_sentence_model()
    sentence_embeddings = sentence_model.encode(chain.cleaned_sentence.to_list(), show_progress_bar=False)

    tsne = TSNE(n_components = 2, n_iter=2000)
    big_pca = PCA(n_components = 50)
    tsne_vals = tsne.fit_transform(big_pca.fit_transform(sentence_embeddings))
    tsne = pd.concat([chain, pd.DataFrame(tsne_vals, columns = ['x_tsne', 'y_tsne'],index=chain.index)], axis = 1)
    return tsne

def clear_df():
    del st.session_state['df']
    
@st.cache
def plot_fig(fig_id,x_tsne,y_tsne,sent_id,xlims,ylims,color_list):
    fig = plt.figure(num=fig_id,figsize=(5,5),dpi=200)
    ax = fig.add_subplot(1,1,1)
    ax.plot(x_tsne[:sent_id+1],y_tsne[:sent_id+1],linewidth=0.2,color='gray',zorder=1)
    ax.scatter(x_tsne[:sent_id+1],y_tsne[:sent_id+1],s=5,color=color_list[:sent_id+1],zorder=2)
    ax.scatter(x_tsne[sent_id:sent_id+1],y_tsne[sent_id:sent_id+1],s=50,marker='*',color='orange',zorder=3)
    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)
    ax.axis('off')
    fig.savefig(f'figures/{fig_id}.png')

def pre_render_images(df,input_sent_id):
    sent_id_options = [min(len(df)-1,max(0,input_sent_id+increment)) for increment in [-500,-100,-10,-1,0,1,10,100,500]]
    x_tsne, y_tsne = df.x_tsne, df.y_tsne
    xmax,xmin = (max(x_tsne)//30+1)*30,(min(x_tsne)//30-1)*30
    ymax,ymin = (max(y_tsne)//30+1)*30,(min(y_tsne)//30-1)*30
    color_list = sns.color_palette('flare',n_colors=int(len(df)*1.2))
    sent_list = []
    fig_production = st.progress(0)
    for fig_id,sent_id in enumerate(sent_id_options):
        fig_production.progress(fig_id+1)
        plot_fig(fig_id,x_tsne,y_tsne,sent_id,[xmin,xmax],[ymin,ymax],color_list)
        sent_list.append(df.cleaned_sentence.to_list()[sent_id])
    return sent_list

max_width = 1500
padding_top = 2
padding_right = 5
padding_bottom = 0
padding_left = 5

define_margins = f"""
<style>
    .appview-container .main .block-container{{
        max-width: {max_width}px;
        padding-top: {padding_top}rem;
        padding-right: {padding_right}rem;
        padding-left: {padding_left}rem;
        padding-bottom: {padding_bottom}rem;
    }}
</style>
"""
hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """
st.markdown(define_margins, unsafe_allow_html=True)
st.markdown(hide_table_row_index, unsafe_allow_html=True)

st.header("Demo: Probing BERT's priors with serial reproduction chains")
tokenizer,model = load_model('bert-base-uncased')
mask_id = tokenizer.encode("[MASK]")[1:-1][0]
input_type = st.sidebar.radio(label='1. Choose the input type',options=('Use one of our example sentences','Use your own initial sentence'))
if input_type=='Use one of our example sentences':
    sentence = st.sidebar.selectbox("Select the inital sentence",
                            ('About 170 campers attend the camps each week.',
                            'She grew up with three brothers and ten sisters.'))
    if sentence=='About 170 campers attend the camps each week.':
        sentence_num = 6
    else:
        sentence_num = 8
    
    st.session_state.df = load_data(sentence_num)
    
else:
    sentence = st.sidebar.text_input('Type down your own sentence here',on_change=clear_df)
    if st.sidebar.button('Run chains'):
        chain = run_chains(tokenizer,model,mask_id,sentence,num_steps=1000)
        st.session_state.df = run_tsne(chain)

if 'df' in st.session_state:
    df = st.session_state.df
    sent_id = st.sidebar.slider(label='2. Select the position in a chain to start exploring',
                                min_value=0,max_value=len(df)-1,value=0)
    
    explore_type = st.sidebar.radio('3. Choose the way to explore',options=['In increments','Flick through samples','autoplay'])
    if explore_type=='autoplay':
        st.write('Play video')
    else:
        if explore_type=='In increments':
            button_labels = ['-500','-100','-10','-1','0','+1','+10','+100','+500']
            increment = st.sidebar.radio(label='select increment',options=button_labels,index=4)
            sent_id += int(increment.replace('+',''))
            sent_id = min(len(df)-1,max(0,sent_id))
        elif explore_type=='Flick through samples':
            sent_id = st.sidebar.number_input(label='step number',value=sent_id)

        x_tsne, y_tsne = df.x_tsne, df.y_tsne
        xmax,xmin = (max(x_tsne)//30+1)*30,(min(x_tsne)//30-1)*30
        ymax,ymin = (max(y_tsne)//30+1)*30,(min(y_tsne)//30-1)*30
        color_list = sns.color_palette('flare',n_colors=int(len(df)*1.2))

        fig = plt.figure(figsize=(5,5),dpi=200)
        ax = fig.add_subplot(1,1,1)
        ax.plot(x_tsne[:sent_id+1],y_tsne[:sent_id+1],linewidth=0.2,color='gray',zorder=1)
        ax.scatter(x_tsne[:sent_id+1],y_tsne[:sent_id+1],s=5,color=color_list[:sent_id+1],zorder=2)
        ax.scatter(x_tsne[sent_id:sent_id+1],y_tsne[sent_id:sent_id+1],s=50,marker='*',color='blue',zorder=3)
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])
        ax.axis('off')

        sentence = df.cleaned_sentence.to_list()[sent_id]
        show_candidates = st.checkbox('Show candidates')
        if show_candidates:
            st.write('click any word to see each candidate with its probability')
            input_sent = tokenizer(sentence,return_tensors='pt')['input_ids']
            decoded_sent = [tokenizer.decode([token]) for token in input_sent[0]]
            cols = st.columns(len(decoded_sent))
            with cols[0]:
                st.write(decoded_sent[0])
            with cols[-1]:
                st.write(decoded_sent[-1])
            for word_id,(col,word) in enumerate(zip(cols[1:-1],decoded_sent[1:-1])):
                with col:
                    if st.button(word):
                        probs = mask_prob(model,mask_id,input_sent,word_id+1)
                        _,df = sample_words(probs, word_id+1, input_sent)
                        st.table(df)
        else:
            if explore_type=='Flick through samples' and input_type=='Use your own initial sentence' and sent_id>0:
                sampled_loc = df.next_sample_loc.to_list()[sent_id-1]
                input_sent = tokenizer(sentence,return_tensors='pt')['input_ids']
                decoded_sent = [tokenizer.decode([token]) for token in input_sent[0]]
                disp_sent_before = '<p style="font-family:san serif; color:Black; font-size: 25px; font-weight:bold">'+' '.join(decoded_sent[1:sampled_loc])
                new_word = f'<span style="color:Red">{decoded_sent[sampled_loc]}</span>'
                disp_sent_after = ' '.join(decoded_sent[sampled_loc+1:-1])+'</p>'
                st.markdown(disp_sent_before+' '+new_word+' '+disp_sent_after,unsafe_allow_html=True)
            else:
                st.markdown(f'<p style="font-family:san serif; color:Black; font-size: 25px; font-weight:bold">{sentence}</p>',unsafe_allow_html=True)
        cols = st.columns([1,2,1])
        with cols[1]:
            st.pyplot(fig)
