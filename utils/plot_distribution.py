#%%
import shelve
import tqdm
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data = shelve.open('/home/aniljegga1/bigdataserver/pubmed_docs/abstracts/pubmed_small.db')
print('total number of paragraphs is', len(list(data.keys())))
keys = list(data.keys())

lens = []

for key in tqdm.tqdm(keys):
    try:
        abstract = data[key]['abstract']
    except:
        print(f'Skipped key {key}')
        continue
    abstract = nltk.sent_tokenize(abstract)
    lens.extend([len(sent.split(' ')) for sent in abstract])
#%%
ax = sns.distplot(lens)
x = np.arange(0, 500, 30)
plt.xticks(x)
plt.show()

#%%
max_lens = []
for key in tqdm.tqdm(keys):
    try:
        abstract = data[key]['abstract']
    except:
        print(f'Skipped key {key}')
        continue
    abstract = nltk.sent_tokenize(abstract)
    try:
        max_len = max([len(sent.split(' ')) for sent in abstract])
    except:
        print(abstract)
    max_lens.append(max_len)
#%%
ax = sns.distplot(max_lens)
x = np.arange(0, 500, 30)
plt.xticks(x)
plt.show()
#%%
from collections import Counter
counts = Counter(max_lens)
print({k: v for k, v in sorted(counts.items(), key=lambda item: item[1])})