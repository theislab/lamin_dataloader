import os
import numpy as np
import pandas as pd
from typing import Dict, List
from lamin_dataloader.utils import normalize
from torch.utils.data import Dataset, default_collate, default_convert, get_worker_info
from abc import ABC, abstractmethod
import random
from pathlib import Path
import re

class Tokenizer(ABC):
    """
    An abstract class for tokenizers.
    
    Args:
        vocabulary (List): A list of feature names.
    """
    
    def __init__(self, vocabulary):
        self.vocabulary = np.array(vocabulary)
    
    @abstractmethod
    def encode(self, items):
        pass
    
    @abstractmethod
    def decode(self, items):
        pass

class GeneIdTokenizer(Tokenizer):
    """
    A Tokenizer subclass that maps gene IDs to tokens.

    Args:
        gene_mapping_path (str): The path to the gene mapping pandas.Series file.
    """
    NOT_FOUND = -1

    def __init__(self, gene_mapping: Dict):
        super().__init__(list(gene_mapping.keys()))
        self.gene_mapping = gene_mapping
        self.reverse_mapping = {v: k for k, v in gene_mapping.items()}
        self.PAD_TOKEN = self.gene_mapping.get('<pad>')

    def encode(self, items):
        return np.array([self.gene_mapping.get(item, self.NOT_FOUND) for item in items])

    def decode(self, items):
        return np.array([self.reverse_mapping.get(item) for item in items])
    

class TokenizedDataset(Dataset):

    def __init__(self, 
                 collection, 
                 tokenizer, 
                 obs_keys=[], 
                 normalization='log1p', 
                 sub_sample_frac=None, 
                 var_column=None):
        super(TokenizedDataset).__init__()
        
        self.collection = collection
        self.tokenizer = tokenizer
        self.normalization = normalization
        self.obs_keys = obs_keys

        if sub_sample_frac is not None:
            # self.collection.subset_data(sub_sample_frac)
            raise NotImplementedError('Subsampling is not implemented yet.')
                
        self.tokenized_vars = []
        for i, var_name in enumerate(self.collection.var_list):
            tokenized_var = self.tokenizer.encode(var_name)
            self.tokenized_vars.append(tokenized_var)
            
        self.masks = []
        self.tokenized_vars_masked = []
        for i, var_name in enumerate(self.collection.var_list):
            mask = self.tokenized_vars[i] != self.tokenizer.NOT_FOUND
            self.tokenized_vars_masked.append(self.tokenized_vars[i][mask])
            assert any(mask), f'dataset {self.collection._path_list[i]} has no token in common with vocabulary.'
            self.masks.append(mask)
        
        
        for i in range(len(self.masks)):
            print(f'Dataset {i}: {self.masks[i].sum()} / {len(self.masks[i])} tokens')

        coverage = []
        for i in range(len(self.masks)):
            coverage.append(self.masks[i].sum() / len(self.masks[i]))
        print(f'coverage macro: {np.mean(coverage)}')
        coverage_micro = sum(np.array(coverage) * np.array(self.collection.n_obs_list)/sum(self.collection.n_obs_list))
        print(f'covarage micro: {coverage_micro}')
        
        
    def __len__(self):
        return len(self.collection)

    def __getitem__(self, idx):
        item = self.collection[idx]
        dataset_id = item['dataset_id']
        mask = self.masks[dataset_id]
                
        tokens = self.tokenized_vars_masked[dataset_id]
        
        values = normalize(item['X'], self.normalization)
        values = values[mask]
                
        return {'tokens': tokens,
                'values': values,
                'dataset_id': dataset_id,
                **{key: item[key] for key in self.obs_keys}
        }

class CustomCollate:
    def __init__(self, 
                 tokenizer, 
                 max_tokens,
                 ):
        self.tokenizer = tokenizer
        self.PAD_TOKEN = tokenizer.PAD_TOKEN
        self.max_tokens = max_tokens
        
        self._rng = None
    
    @property
    def rng(self):
        if self._rng is None:
            self._rng = np.random.default_rng(42)
        return self._rng

    def resize_and_pad(self, item, max_tokens):
        tokens, values = item['tokens'], item['values']
        
        context_size = min(len(tokens), max_tokens)
        tokens, values = tokens[:context_size], values[:context_size]

        sorted_indices = np.argsort(values)[::-1]
        tokens, values = tokens[sorted_indices], values[sorted_indices]
        
        pad = max(max_tokens - context_size, 0)
        tokens = np.pad(tokens, (0, pad), mode='constant', constant_values=self.PAD_TOKEN)
        values = np.pad(values, (0, pad), mode='constant', constant_values=0)
        return {'tokens': tokens, 'values': values}


    def __call__(self, batch):
        n_tokens = len(batch[0]['tokens'])
        permute = self.rng.permutation(n_tokens)
        batch_ = [{'tokens': item['tokens'][permute], 
                          'values': item['values'][permute], 
                          } for item in batch]
        
                
        max_lenght = max([len(item['tokens']) for item in batch_])
        
        max_lenght = min(max_lenght, self.max_tokens - 1)

        batch_ = [self.resize_and_pad(item, max_lenght) for item in batch_]
        
        tokens, values = [item['tokens'] for item in batch_], [item['values'] for item in batch_]
    
        return {'tokens': default_collate(tokens),
                'values': default_collate(values),
                **{key: default_collate([item[key] for item in batch]) for key in batch[0].keys() if key not in ['tokens', 'values']}
        }
