import pymongo as pm
import pandas as pd
import json

# this auth.json file contains credentials
with open('auth.json') as f :
    auth = json.load(f)
user = auth['user']
pswd = auth['password']

# initialize mongo connection
conn = pm.MongoClient('mongodb://{}:{}@127.0.0.1'.format(user, pswd))

# get database for this project
db = conn['telephone-game']

# get stimuli collection from this database
print('possible collections include: ', db.collection_names())
stim_coll = db['stimuli']

# empty stimuli collection if already exists
# (note this destroys records of previous games)
if stim_coll.count() != 0 :
    stim_coll.drop()

# Loop through evidence and insert into collection
sets = {
    'short' : pd.read_csv('./Samples/wiki/12TokenSents.csv'),
    'medium' : pd.read_csv('./Samples/wiki/21TokenSents.csv'),
    'long' : pd.read_csv('./Samples/wiki/37TokenSents.csv')
}

for dataset in ['short', 'medium', 'long'] :
    set = sets[dataset]
    for i, row in set.iterrows() :
        packet = {
            'length' : dataset, 'sample_id' : row['sample_id'],
            'folder_id' : row['folder_id'], 'sent_id' : row['sent_id'],
            'sentence' : row['sentence'], 'prob' : row['prob_1'],
            'numGames': 0, 'games' : []
        }
        stim_coll.insert_one(packet)

print('checking one of the docs in the collection...')
print(stim_coll.find_one())
