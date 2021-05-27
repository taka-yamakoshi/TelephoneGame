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

dataset = pd.read_csv('./stimuli/grouped_stims_for_mongo_cleaned.csv')

for group_name, group in dataset.groupby('group_id') :
    trials = []
    for row_i, row in group.iterrows() :
        trials.append(row.to_dict())

    packet = {
        'trials' : trials,
        'set_id' : group_name,
        'numGames': 0,
        'games' : []
    }
    stim_coll.insert_one(packet)

print('checking one of the docs in the collection...')
print(stim_coll.find_one())
