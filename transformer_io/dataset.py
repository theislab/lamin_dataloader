import os
from typing import Dict, List
from transformer_io.data.utils import normalize

import numpy as np
from torch.utils.data import Dataset, default_collate

from abc import ABC, abstractmethod

import lamindb as ln
#todo: no hardcode names
ln.connect('mojtababahrami/lamindb-storage')

from .mapped_collection import CustomMappedCollection

np.random.seed(42)


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

    def encode(self, items):
        return np.array([self.gene_mapping.get(item, self.NOT_FOUND) for item in items])

    def decode(self, items):
        return np.array([self.reverse_mapping.get(item) for item in items])

class TokenizedDataset(Dataset):

    def __init__(self, dataset_path, filenames, tokenizer, n_tokens, obs_keys=[], normalization='log1p', sub_sample_frac=None, var_column=None):
        super(TokenizedDataset).__init__()
        
        path_list = [os.path.join(dataset_path, file) for file in filenames]
            
        self.mc = CustomMappedCollection(path_list, layers_keys="X", obs_keys=obs_keys, join=None, encode_labels=True, parallel=True)
        if sub_sample_frac is not None:
            self.mc.subset_data(sub_sample_frac)

        self.path_list = path_list
        self.obs_keys = obs_keys
        self.tokenizer = tokenizer
        self.normalization = normalization
        self.n_tokens = n_tokens

        self.tokenized_vars = []
        for i, var_name in enumerate(self.mc.var_list):
            tokenized_var = self.tokenizer.encode(var_name)
            self.tokenized_vars.append(tokenized_var)
            
        self.masks = []
        for i, var_name in enumerate(self.mc.var_list):
            mask = self.tokenized_vars[i] != self.tokenizer.NOT_FOUND
            assert any(mask), f'dataset {self.path_list[i]} has no token in common with vocabulary.'
            self.masks.append(mask)
        
        if len(self.masks) < 5:
            for i in range(len(self.masks)):
                print(f'Dataset {filenames[i]}: {self.masks[i].sum()} / {len(self.masks[i])} tokens')
        coverage = []
        for i in range(len(self.masks)):
            coverage.append(self.masks[i].sum() / len(self.masks[i]))
        print(f'coverage macro: {np.mean(coverage)}')
        coverage_micro = sum(np.array(coverage) * np.array(self.mc.n_obs_list)/sum(self.mc.n_obs_list))
        print(f'covarage micro: {coverage_micro}')
        
        
        
    def __len__(self):
        return len(self.mc)

    def __getitem__(self, idx):
        item = self.mc[idx]
        _store_idx = item['_store_idx']
        mask = self.masks[_store_idx]
        
        var_names = self.mc.var_list[_store_idx]
        var_names = var_names[mask]
        
        tokens = self.tokenized_vars[_store_idx]
        tokens = tokens[mask]
        
        values = normalize(item['X'], self.normalization)
        values = values[mask]
        
        n_tokens = min(self.n_tokens, len(tokens))
        selected_vars = np.random.choice(range(len(tokens)), n_tokens, replace=False)
        tokens = tokens[selected_vars]
        values = values[selected_vars]

        return {'tokens': tokens,
                'values': values,
                'storage': _store_idx,
                **{key: item[key] for key in self.obs_keys}
        }
        
        

def custom_collate(batch):
    if isinstance(batch, Dict):
        return {key: default_collate(batch[key]) for key in batch.keys()}
    else:
        return default_collate(batch)
    

        

