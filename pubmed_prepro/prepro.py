import json
import gzip
import pickle
import shelve
import os
from multiprocessing import Pool, cpu_count
import xml.etree.ElementTree as ET
from tqdm import tqdm
import time
import torch
'''
Process each pubmed xml file 
> Get abstract and pubmedid
> store them shelve db
'''

start_time = time.time()


with open('/home/aniljegga1/bigdataserver/pubmed_docs/abstracts/filtered_pubmedids.pkl', 'rb') as fin:
    pubmed_ids = pickle.load(fin)

def read_abstracts(infile):
    db = shelve.open('/home/aniljegga1/bigdataserver/pubmed_docs/abstracts/pubmed_small/pubmed.db')

    root = ET.parse(infile).getroot()
    for article in root.iter(tag='PubmedArticle'):
        pmid = [pmid.text for pmid in article.iter(tag='PMID')][0]
        pmid = int(pmid)
        if pmid not in pubmed_ids: continue
        for abstract in article.iter(tag='Abstract'):
            text = ''
            for elem in abstract.iter(tag='AbstractText'):
                if elem.text is None:
                    continue
                text += elem.text + ' '
            db[str(pmid)] = {'abstract':text}

    db.close()

def process_xmls():
    xml_dir = '/home/aniljegga1/bigdataserver/pubmed_docs/'
    xml_files = os.listdir(xml_dir)
    xml_files = [xml_dir+file for file in xml_files if file.endswith('.xml')]

    for xml_file in tqdm(xml_files):
        read_abstracts(xml_file)

    complete_time = (time.time() - start_time)/60
    print(f'Processing completed in {complete_time} mins')


process_xmls()

# keys = list(db.keys())[:10]
# print(keys)
# for key in keys:
#     print(key)
#     try:
#         x = db[key]
#         print(x)
#     except:
#         print(f'skipped key {key}')
#         continue

