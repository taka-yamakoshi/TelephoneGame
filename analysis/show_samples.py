import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go

df = pd.read_csv('../behavioral_experiment/data/averaged_responses.csv')
col1, col2 = st.columns(2)
with col1:
    sort_option = st.selectbox('sort by',
                        ('phase', 'sent_length','prob','avg_response'))
with col2:
    sort_dir = st.selectbox('ascending or descending?',
                        ('ascending', 'descending'))

sort_dir = sort_dir=='ascending'
new_df = df.sort_values(by=sort_option,ascending=sort_dir).\
            loc[lambda d: d['source']=='bert'].\
            drop(columns=['step','source','sentence_id','initial_state_prob_class']).\
            reindex(columns=['phase', 'sent_length','sentence','prob','avg_response'])

max_width = 1500
padding_top = 2
padding_right = 0
padding_bottom = 0
padding_left = 0

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
st.table(new_df)
