from typing import List, Union
import collections
import re

def get_time_machine_dataset():
    with open("./dataset/time_machine.txt", "r") as f:
        lines = f.readlines()
    return lines

def tokenize(lines, token='char'):  #@save
    """Split text lines into word or character tokens."""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('ERROR: unknown token type: ' + token)

class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None: tokens = []
        self.reserved_tokens = [] if reserved_tokens is None else reserved_tokens
        

        self.idx_to_token = ["<unk>"] + self.reserved_tokens
        self.token_to_idx = {self.idx_to_token[idx]: idx for idx in range(len(self))}

        self._token_freqs = self._get_token_freqs(tokens)
        for token, freq in self._token_freqs:
            if freq < min_freq: break
                
            if token not in self.idx_to_token:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self)-1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, 
        tokens: Union[str, List[str]]
    ) -> Union[int, List[int]]:

        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def _get_token_freqs(self, tokens_list: List[List[str]]):
        tokens = []
        for line in tokens_list:
            for token in line:
                tokens.append(token)
        tokens_count = collections.Counter(tokens).items()
        return sorted(tokens_count, key=lambda x:x[1], reverse=True)

    def to_tokens(self, idx):
        if isinstance(idx, int):
            return self.idx_to_token[idx]
        
        return [self.idx_to_token[i] for i in idx]

    @property
    def token_freqs(self):
        return self._token_freqs
    
    @property
    def unk(self):
        return self.token_to_idx['<unk>']
    
    @property
    def get_vocab(self):
        return self.idx_to_token

    @property
    def get_reserved_tokens(self):
        return self.reserved_tokens

import tensorflow as tf

class DataLoader:
    def __init__(self,
        batch_size,
        num_steps,
        tokens: List[List[str]]  # tokenized list of sentences
    ) -> None:
        self.batch_size = batch_size
        self.num_steps  = num_steps
        self.vocab      = Vocab(tokens)
        self.corpus     = [token for line in tokens for token in line]
        self.data_iter_fn = self.seq_data_loader

    def seq_data_loader(self):

        num_tokens = (len(self.corpus)-1) // self.batch_size * self.batch_size

        X = tf.constant( self.corpus[0:num_tokens] )
        Y = tf.constant( self.corpus[1:1+num_tokens] )

        Xs = tf.reshape(X, (self.batch_size, -1))
        Ys = tf.reshape(Y, (self.batch_size, -1))

        """ for i in range(X.shape[1] - self.num_steps + 1):
            yield X[:, i: i+self.num_steps], Y[:, i: i+self.num_steps] """

        num_batches = Xs.shape[1] // self.num_steps
        for i in range(0, num_batches * self.num_steps, self.num_steps):
            X = Xs[:, i: i + self.num_steps]
            Y = Ys[:, i: i + self.num_steps]
            yield X, Y
    
    def __iter__(self):
        return self.data_iter_fn()




if __name__ == '__main__':

    try:
        import string
        tokens = list(string.ascii_lowercase)
        vocab = Vocab(tokens = tokens)

        print(
            vocab['t'], 
            vocab[['t', 'a', 's']], 
            vocab.to_tokens(4), 
            len(vocab)
        )
   
    except Exception as e:
        print(e)
