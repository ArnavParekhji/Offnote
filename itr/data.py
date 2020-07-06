
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import pandas as pd
import numpy as np

from pathlib import Path

def split_data(file_path, destination):

    data = pd.read_csv(file_path, sep='\t', header=None)[[0, 1]]

    mask = np.random.rand(data.shape[0]) < 0.8
    train = data[mask]
    valid = data[~mask]

    train.to_csv(Path(destination) / 'train', header=False, index=False)
    valid.to_csv(Path(destination) / 'valid', header=False, index=False)



class PadSequence:
    
    def __init__(self, src_padding_value, tgt_padding_value):
        self.src_padding_value = src_padding_value
        self.tgt_padding_value = tgt_padding_value
    
    def __call__(self, batch):
        
        x = [s[0] for s in batch]
        x = pad_sequence(x, 
                         batch_first=True, 
                         padding_value=self.src_padding_value)

        y = [s[1] for s in batch]
        y = pad_sequence(y, 
                         batch_first=True, 
                         padding_value=self.tgt_padding_value)

        return x, y


class IndicDataset(Dataset):
  
    def __init__(self, 
                 src_tokenizer,
                 tgt_tokenizer,
                 filepath,
                 is_train=True):
        filepath += 'train' if is_train else 'valid'
        self.df = pd.read_csv(filepath, header=None)

        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        y, x = self.df.loc[index]
 
        #tokenize into integer indices
        x = self.src_tokenizer.convert_tokens_to_ids(self.src_tokenizer.tokenize(x))
        y = self.tgt_tokenizer.convert_tokens_to_ids(self.tgt_tokenizer.tokenize(y))

        #add special tokens to target
        y = [self.tgt_tokenizer.bos_token_id] + y + [self.tgt_tokenizer.eos_token_id]

        return torch.LongTensor(x), torch.LongTensor(y)
