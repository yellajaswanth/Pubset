import json
import pickle
import torch
from pytorch_pretrained_bert_inset import BertTokenizer, BertModel
from torch.nn.utils.rnn import pad_sequence
import tqdm
import shelve
import nltk

tokenizer = BertTokenizer.from_pretrained('biobert-base')
model_bert = BertModel.from_pretrained('biobert-base', state_dict=torch.load('auto_log/GPT2.2e-05.64.1gpu.2021-02-08131841/BERT-pretrain-5-step-4500.pkl')).cuda()

data = shelve.open('dataset/pubmed_small/pubmed.db')
print('total number of paragraphs is', len(list(data.keys())))

keys = list(data.keys())
model_bert.eval()
with torch.no_grad():
    for key in tqdm.tqdm(keys):
        try:
            abstract = data[key]['abstract']
        except:
            # print(f'Skipped key {key}')
            continue
        abstract = nltk.sent_tokenize(abstract)
        if len(abstract) > 6:
            ids_unpad = []
            for i in range(len(abstract)):
                ids_unpad.append(torch.tensor([101] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(abstract[i])) + [102], dtype=torch.long))
            ids = pad_sequence(ids_unpad, batch_first=True, padding_value=0).cuda()
            x = torch.zeros(ids.size(0), 768).cuda()
            if ids.size(-1) < 52:
                encoded_layers, _ = model_bert(ids, torch.zeros_like(ids), 1 - (torch.zeros_like(ids) == ids).type(torch.uint8), False)
                x[:, :] = encoded_layers[:, 0, :]
                value = data[key]
                value['bert'] = x.half()
                data[key] = value
        torch.cuda.empty_cache()

data.close()