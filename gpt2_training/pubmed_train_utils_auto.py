from gpt2_training.train_utils_auto import *
import shelve
import json
import gzip

class PubmedDynamicBatchingLoader(object):
    def __init__(self, db_path, batch_size, max_seq_length, is_train):
        self.db_path = db_path
        self.db = shelve.open(f'{self.db_path}', 'r')
        self.bs = batch_size
        self.max_seq_length = max_seq_length
        self.train = is_train
        # self.num_examples = self.get_len(self.db)

    def __iter__(self, epoch=1):
        if epoch > 0:
            for epoch in range(epoch):
                yield from self._iter_epoch()
        else:
            while True:
                yield from self._iter_epoch()

    def __len__(self):
        raise NotImplementedError()

    def _iter_epoch(self):
        try:
            examples = []
            for key in self.db:
                chunk = json.loads(gzip.decompress(self.db[key]).decode('utf-8'))                
                for line_bert, line_gpt in chunk:
                    if (len(line_bert) < self.max_seq_length) and (len(line_gpt) < self.max_seq_length):
                        examples.append({'input_ids_bert' : [101] + line_bert + [102], 'input_ids_gpt': [50256] + line_gpt + [50256]})
                        if len(examples) == self.bs:                
                            batch = self._batch_feature(examples)
                            examples = []
                            yield batch
        except StopIteration:
            pass

    def _batch_feature(self, features):
        input_ids_bert = pad_sequence([torch.tensor(f['input_ids_bert'], dtype=torch.long) for f in features], batch_first=True, padding_value=0)
        input_ids_gpt = pad_sequence([torch.tensor(f['input_ids_gpt'], dtype=torch.long) for f in features], batch_first=True, padding_value=0)
        lm_labels = pad_sequence([torch.tensor(f['input_ids_gpt'], dtype=torch.long) for f in features], batch_first=True, padding_value=-1)
        return (input_ids_bert, input_ids_gpt, lm_labels)

    def get_len(self, db):
        raise NotImplementedError()