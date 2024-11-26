import os
import numpy as np

from typing import Dict, List
from transformer_io.utils import normalize
from torch.utils.data import Dataset, default_collate
from abc import ABC, abstractmethod
import random


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

    def __init__(self, collection, tokenizer, max_tokens, min_tokens=1, obs_keys=[], normalization='log1p', gene_sampling_strategy='random', sub_sample_frac=None, var_column=None):
        super(TokenizedDataset).__init__()
        
        self.collection = collection
        self.tokenizer = tokenizer
        self.normalization = normalization
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.obs_keys = obs_keys
        self.gene_sampling_strategy = gene_sampling_strategy
        assert self.gene_sampling_strategy in ['random', 'random-nonzero'], 'gene_sampling_strategy must be either "random" or "random-nonzero"'
        

        if sub_sample_frac is not None:
            # self.collection.subset_data(sub_sample_frac)
            raise NotImplementedError('Subsampling is not implemented yet.')
                
        self.tokenized_vars = []
        for i, var_name in enumerate(self.collection.var_list):
            tokenized_var = self.tokenizer.encode(var_name)
            self.tokenized_vars.append(tokenized_var)
            
        self.masks = []
        for i, var_name in enumerate(self.collection.var_list):
            mask = self.tokenized_vars[i] != self.tokenizer.NOT_FOUND
            assert any(mask), f'dataset {self.path_list[i]} has no token in common with vocabulary.'
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
        
        var_names = self.collection.var_list[dataset_id]
        var_names = var_names[mask]
        
        tokens = self.tokenized_vars[dataset_id]
        tokens = tokens[mask]
        
        values = normalize(item['X'], self.normalization)
        values = values[mask]
        
        n_tokens = len(tokens)
        selected_vars = np.random.choice(range(len(tokens)), n_tokens, replace=False)
        tokens, values = tokens[selected_vars], values[selected_vars]
        
        tokens_1, tokens_2 = tokens[:n_tokens//2], tokens[n_tokens//2:]
        values_1, values_2 = values[:n_tokens//2], values[n_tokens//2:]
        
        if self.gene_sampling_strategy == 'random-nonzero':
            nonzero_mask_1 = values_1 > 0
            nonzero_mask_2 = values_2 > 0
            tokens_1, values_1 = tokens_1[nonzero_mask_1], values_1[nonzero_mask_1]
            tokens_2, values_2 = tokens_2[nonzero_mask_2], values_2[nonzero_mask_2]
            
            context_size_1 = random.randint(min(len(tokens_1), self.min_tokens), min(len(tokens_1), self.max_tokens))
            context_size_2 = random.randint(min(len(tokens_2), self.min_tokens), min(len(tokens_2), self.max_tokens))
            
            tokens_1, tokens_2 = tokens_1[:context_size_1], tokens_2[:context_size_2]
            values_1, values_2 = values_1[:context_size_1], values_2[:context_size_2]

            assert not np.any(np.isin(tokens_1, tokens_2)), "tokens_1 and tokens_2 should not have shared tokens"
            
            pad_1 = max(self.max_tokens - context_size_1, 0)
            pad_2 = max(self.max_tokens - context_size_2, 0)

            tokens_1 = np.pad(tokens_1, (0, pad_1), mode='constant', constant_values=self.tokenizer.PAD_TOKEN)
            values_1 = np.pad(values_1, (0, pad_1), mode='constant', constant_values=0)
            tokens_2 = np.pad(tokens_2, (0, pad_2), mode='constant', constant_values=self.tokenizer.PAD_TOKEN)
            values_2 = np.pad(values_2, (0, pad_2), mode='constant', constant_values=0)
            
        else:
            tokens_1, tokens_2 = tokens_1[:self.max_tokens], tokens_2[:self.max_tokens]
            values_1, values_2 = values_1[:self.max_tokens], values_2[:self.max_tokens]
            
        return {'tokens_1': tokens_1,
                'values_1': values_1,
                'tokens_2': tokens_2,
                'values_2': values_2,
                'dataset_id': dataset_id,
                **{key: item[key] for key in self.obs_keys}
        }
        
        

def custom_collate(batch):
    if isinstance(batch, Dict):
        return {key: default_collate(batch[key]) for key in batch.keys()}
    else:
        return default_collate(batch)
    

        

