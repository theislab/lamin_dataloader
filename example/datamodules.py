import os
from typing import Dict, List
import lightning as L
from torch.utils.data import DataLoader
from transformer_io.dataset import TokenizedDataset, Tokenizer, custom_collate
from transformer_io.collections import CustomMappedCollection

class MappedCollectionDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_path: str,
        split: Dict,
        tokenizer: Tokenizer,
        columns: List[str],
        normalization: str = 'log1p',
        dataset_kwargs: Dict = {},
        dataloader_kwargs: Dict = {},
    ):
        super().__init__()
        self.dataloader_kwargs = dataloader_kwargs

        dataset_kwargs_shared = {'obs_keys': columns,
                                 'tokenizer': tokenizer, 
                                 'normalization': normalization}


        if 'train' in split and split['train'] is not None and 'train' in dataset_kwargs:
            path_list = [os.path.join(dataset_path, file) for file in split['train']]
            collection = CustomMappedCollection(path_list, layers_keys="X", obs_keys=columns, join=None, encode_labels=True, parallel=True)
            self.train_dataset = TokenizedDataset(**{'collection': collection, **dataset_kwargs_shared, **dataset_kwargs['train']})
        if 'val' in split and split['val'] is not None and 'val' in dataset_kwargs:
            path_list = [os.path.join(dataset_path, file) for file in split['val']]
            collection = CustomMappedCollection(path_list, layers_keys="X", obs_keys=columns, join=None, encode_labels=True, parallel=True)
            self.val_dataset = TokenizedDataset(**{'collection': collection, **dataset_kwargs_shared, **dataset_kwargs['val']})


    def _get_dataloader(self, dataset, dataloader_kwargs, stage):
        assert stage in ['train', 'val']
        
        dataloader = DataLoader(dataset,
                                worker_init_fn=dataset.collection.torch_worker_init_fn,
                                collate_fn=custom_collate,
                                **dataloader_kwargs)
        return dataloader
        
    def train_dataloader(self):
        dataloader_kwargs = self.dataloader_kwargs['train']
        dataloader = self._get_dataloader(self.train_dataset, dataloader_kwargs, 'train')
        return dataloader

    def val_dataloader(self):
        dataloader_kwargs = self.dataloader_kwargs['val']
        dataloader = self._get_dataloader(self.val_dataset, dataloader_kwargs, 'val')
        return dataloader
