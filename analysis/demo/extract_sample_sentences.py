# This is a sample code to extract chains for the t-sne visualization
# Please replace PATH_TO_THE_CSV_FILE with your path to the csv file containing the generated samples
import pandas as pd
fname = 'PATH_TO_THE_CSV_FILE'
df = pd.read_csv(fname)
df.loc[lambda d:(d['chain_num']==0)&((d['sentence_num']==2)|(d['sentence_num']==6)|(d['sentence_num']==8))].\
        to_csv(f'sample_sentences.csv',index=False)
