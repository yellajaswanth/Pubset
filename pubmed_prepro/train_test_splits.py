#%%
import shelve
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split

db = shelve.open('/home/aniljegga1/bigdataserver/pubmed_docs/abstracts/pubmed_small/pubmed.db', 'r')

keys = list(db.keys())

all_indices = []

for key in tqdm(keys):
    if 'bert' in db[key]:
        num_sents = len(db[key]['bert'])
        all_indices.extend([(key, i) for i in range(num_sents-6)])

train, test = train_test_split(all_indices, test_size=0.25)
train, val = train_test_split(train, test_size=0.15)

#%%

with open('/home/aniljegga1/bigdataserver/pubmed_docs/abstracts/pubmed_small/all_indices.json', 'w') as fin:
    json.dump(all_indices, fin)

with open('/home/aniljegga1/bigdataserver/pubmed_docs/abstracts/pubmed_small/train_indices.json', 'w') as fin:
    json.dump(train, fin)

with open('/home/aniljegga1/bigdataserver/pubmed_docs/abstracts/pubmed_small/test_indices.json', 'w') as fin:
    json.dump(test, fin)

with open('/home/aniljegga1/bigdataserver/pubmed_docs/abstracts/pubmed_small/val_indices.json', 'w') as fin:
    json.dump(val, fin)
    

