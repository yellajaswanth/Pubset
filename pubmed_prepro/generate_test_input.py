import json
import shelve
import nltk

db = shelve.open('/home/aniljegga1/bigdataserver/pubmed_docs/abstracts/pubmed_small/pubmed.db', 'r')

with open('/home/aniljegga1/bigdataserver/pubmed_docs/abstracts/pubmed_small/test_indices.json', 'r') as fin:
    test_indices = json.load(fin)


input = []
count = 0
for indices in test_indices:
    if count == 10: break
    # if indices[1] == 0:
    abstract = db[indices[0]]
    abstract = nltk.sent_tokenize(abstract['abstract'])
    abstract = abstract[:7]
    abstract = ' '.join(abstract[:3]) + '\t' + ' '.join(abstract[4:]) + '\n'
    with open('pubmed_input.txt', 'a') as finput:
        finput.write(abstract)
        count+=1

print('Write input completed!')