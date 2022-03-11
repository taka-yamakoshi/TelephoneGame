import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache
def load_data(sentence_num):
    df = pd.read_csv('movies/tsne_out.csv')
    df = df.loc[lambda d: (d['sentence_num']==sentence_num)&(d['iter_num']<1000)]
    return df


sentence = st.sidebar.selectbox("Select the inital sentence",
                        ('About 170 campers attend the camps each week.',
                        'She grew up with three brothers and ten sisters.'))

if sentence=='About 170 campers attend the camps each week.':
    sentence_num = 6
else:
    sentence_num = 8

df = load_data(sentence_num)
sent_id = st.sidebar.slider(label='move the slider to see how the sentence moves around',min_value=0,max_value=len(df)-1,value=1)
iter_id = sent_id//10
step_id = sent_id%10

col1, col2 = st.columns(2)
with col1:
    iter_id = st.sidebar.number_input(label='iteration number',value=iter_id)
with col2:
    step_id = st.sidebar.number_input(label='step number',value=step_id)

sent_id = 10*iter_id+step_id

x_tsne, y_tsne = df.x_tsne, df.y_tsne
color_list = sns.color_palette('hls',n_colors=len(df))
fig,ax = plt.subplots(1,1,figsize=(5,5))
ax.plot(x_tsne[:sent_id+1],y_tsne[:sent_id+1],linewidth=0.2,color='gray')
ax.scatter(x_tsne[:sent_id+1],y_tsne[:sent_id+1],s=5,color=color_list[:sent_id+1])
ax.scatter(x_tsne[sent_id:sent_id+1],y_tsne[sent_id:sent_id+1],s=50,marker='*',color='k')
if sentence_num==6:
    xlim = [-175,150]
    ylim = [-100,175]
elif sentence_num==8:
    xlim = [-175,175]
    ylim = [-175,200]
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.axis('off')

st.subheader(df.cleaned_sentence.to_list()[sent_id])
st.pyplot(fig)
