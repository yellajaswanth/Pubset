
import gzip
import math
import random
import json
import torch
from torch.utils.data import DataLoader, Sampler, Dataset
from torch.nn.utils.rnn import pad_sequence

from utils.prepro_auto import InputFeatures

import shelve


class BucketSampler(Sampler):
    def __init__(self, lens, bucket_size, batch_size, droplast=False, shuffle=True):
        self._lens = lens
        self._batch_size = batch_size
        self._bucket_size = bucket_size
        self._droplast = droplast
        self._shuf = shuffle

    def __iter__(self):
        ids = list(range(len(self._lens)))
        if self._shuf:
            random.shuffle(ids)
        buckets = [sorted(ids[i:i+self._bucket_size],
                          key=lambda i: self._lens[i], reverse=True)
                   for i in range(0, len(ids), self._bucket_size)]
        batches = [bucket[i:i+self._batch_size]
                   for bucket in buckets
                   for i in range(0, len(bucket), self._batch_size)]
        if self._droplast:
            batches = [batch for batch in batches
                       if len(batch) == self._batch_size]
        if self._shuf:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        bucket_sizes = ([self._bucket_size]
                        * (len(self._lens) // self._bucket_size)
                        + [len(self._lens) % self._bucket_size])
        if self._droplast:
            return sum(s//self._batch_size for s in bucket_sizes)
        else:
            return sum(math.ceil(s/self._batch_size) for s in bucket_sizes)


class GPT2FeatureDataset(Dataset):
    def __init__(self, features, max_len=None):
        self.features = features
        self.max_len = max_len  # this max_len do truncate

    def __getitem__(self, i):
        feat_dict = self.features[i]
        feat = InputFeatures(**feat_dict)
        return feat

    def __len__(self):
        return len(self.features)

    @staticmethod
    def collate(features):
        input_ids_bert = pad_sequence([torch.tensor(f.input_ids_bert, dtype=torch.long) for f in features], batch_first=True, padding_value=0)
        input_ids_gpt = pad_sequence([torch.tensor(f.input_ids_gpt, dtype=torch.long) for f in features], batch_first=True, padding_value=0)
        lm_labels = pad_sequence([torch.tensor(f.input_ids_gpt, dtype=torch.long) for f in features], batch_first=True, padding_value=-1)
        return (input_ids_bert, input_ids_gpt, lm_labels)


class BucketingDataLoader(object):
    """ this loads shelve db chunks and then convert to mini-batch loader"""
    def __init__(self, db_path, batch_size, max_seq_length, bucket=100, shuffle=True):
        self.db_path = db_path
        self.db = shelve.open(f'{self.db_path}', 'r')
        self.batch_size = batch_size
        self.max_len = max_seq_length
        self.bucket_size = bucket * batch_size
        self.shuffle = shuffle
        self.db_length = self._get_db_size()
        
    def _get_keys(self):
        keys = list(self.db.keys())
        return keys
    
    def _get_db_size(self):
        keys = self._get_keys()
        db_len_ = 0
        if self.shuffle:
            random.shuffle(keys)
        for index, key in enumerate(keys):
            print(f'loading chunk: {index+1}/{len(keys)} chunks')
            chunk = json.loads(gzip.decompress(self.db[key]).decode('utf-8'))
            # discard long examples
            trunc_chunk = []
            lens = []

            for bert_tok, gpt_tok in chunk:
                if (len(bert_tok) < self.max_len) and (len(gpt_tok) < self.max_len):
                    trunc_chunk.append({'input_ids_bert' : [101] + bert_tok + [102], 'input_ids_gpt': [50256] + gpt_tok + [50256]})
                    lens.append(len(bert_tok))
            dataset = GPT2FeatureDataset(trunc_chunk, self.max_len)
            sampler = BucketSampler(lens, self.bucket_size, self.batch_size, droplast=True, shuffle=self.shuffle)
            loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, collate_fn=GPT2FeatureDataset.collate)
            db_len_ += len(loader)

        return db_len_
    
    def __iter__(self):
        keys = self._get_keys()
        if self.shuffle:
            random.shuffle(keys)
        for index, key in enumerate(keys):
            print(f'loading chunk: {index+1}/{len(keys)} chunks')
            chunk = json.loads(gzip.decompress(self.db[key]).decode('utf-8'))
            # discard long examples
            trunc_chunk = []
            lens = []

            for bert_tok, gpt_tok in chunk:
                if (len(bert_tok) < self.max_len) and (len(gpt_tok) < self.max_len):
                    trunc_chunk.append({'input_ids_bert' : [101] + bert_tok + [102], 'input_ids_gpt': [50256] + gpt_tok + [50256]})
                    lens.append(len(bert_tok))
            dataset = GPT2FeatureDataset(trunc_chunk, self.max_len)
            sampler = BucketSampler(lens, self.bucket_size, self.batch_size, droplast=True, shuffle=self.shuffle)
            loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, collate_fn=GPT2FeatureDataset.collate)
            yield from loader

    def __len__(self):
        return self.db_length

    def __del__(self):
        self.db.close()


class DistributedBucketingDataLoader(BucketingDataLoader):
    """ distributed version """

    def __init__(self, rank, num_replica, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank = rank
        # print(80*"***")
        # print(self.rank)
        self.num_replica = num_replica

    def _get_keys(self):
        keys = list(self.db.keys())[self.rank::self.num_replica]
        return keys


def test():
    from tqdm import tqdm
    device = torch.device('cuda')
    loader = BucketingDataLoader('dataset/db/pubmed_train.db', 32, 32)
    print(f'num_examples: {loader.db_length}')
    print(f'num_batches: {len(loader)}')
    for *batch, _, _ in tqdm(loader):
        for t in batch:
            t = t.to(device)


if __name__ == '__main__':
    # import sys
    # db_path = sys.argv[1]
    test()
