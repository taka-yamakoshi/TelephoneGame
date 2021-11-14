import numpy as np
import csv
from CountFreq import TokenizerSetUpNew

if __name__ == '__main__':
    ##Set up the spacy
    nlp = TokenizerSetUpNew()

    ## Load the data
    with open('../behavioral_experiment/data/averaged_responses.csv','r') as f:
        reader = csv.reader(f)
        file = [row for row in reader]
    head = file[0]
    text = file[1:]
    sentences = [row[head.index('sentence')] for row in text]

    with open('../behavioral_experiment/data/features.csv','w') as f:
        writer = csv.writer(f)
        new_head = head+['pos_list','tag_list','dep_list','dep_dist','vec']
        writer.writerow(new_head)
        assert len(sentences)==len(text)
        for sentence,line in zip(sentences,text):
            doc = nlp(sentence)
            pos_list = [token.pos_ for token in doc]
            tag_list = [token.tag_ for token in doc]
            dep_list = [token.dep_ for token in doc]
            dep_dist = np.sum([abs(token_pos-token.head.i) for token_pos,token in enumerate(doc)])
            vec = doc.vector
            new_line = line+[pos_list,tag_list,dep_list,dep_dist,vec]
            writer.writerow(new_line)
