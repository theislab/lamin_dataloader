import os

from typing import Dict, List
import lightning as L
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler
from transformer_io.dataset import TokenizedDataset, Tokenizer, custom_collate
from transformer_io.collections import MappedCollection
from transformer_io.samplers import WithinStorageSampler
from lightning.fabric.utilities.distributed import DistributedSamplerWrapper



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
            collection = MappedCollection(path_list, layers_keys="X", obs_keys=columns, join=None, encode_labels=True, parallel=True)
            self.train_dataset = TokenizedDataset(**{'collection': collection, **dataset_kwargs_shared, **dataset_kwargs['train']})
        
        if 'val' in split and split['val'] is not None and 'val' in dataset_kwargs:
            path_list = [os.path.join(dataset_path, file) for file in split['val']]
            collection = MappedCollection(path_list, layers_keys="X", obs_keys=columns, join=None, encode_labels=True, parallel=True)
            self.val_dataset = TokenizedDataset(**{'collection': collection, **dataset_kwargs_shared, **dataset_kwargs['val']})


    def _get_dataloader(self, dataset, dataloader_kwargs, stage):
        assert stage in ['train', 'val']
        
        num_replicas = dist.get_world_size() if torch.distributed.is_initialized() else 1
        within_batch_sampling = dataloader_kwargs.pop('within_dataset_sampling')
        batch_size = dataloader_kwargs.pop('batch_size') // num_replicas
        shuffle = dataloader_kwargs.pop('shuffle')
        drop_last = dataloader_kwargs.pop('drop_last')
        assert drop_last == True, 'drop_last must be True during training and validation'
        assert shuffle == True, 'shuffle must be True during training and validation'
        if within_batch_sampling:
            sampler = WithinStorageSampler(dataset.collection.n_obs_list, batch_size, num_replicas, shuffle=shuffle, drop_last=drop_last)
        else:
            sampler = RandomSampler(dataset)
        
        if torch.distributed.is_initialized():
            sampler = DistributedSamplerWrapper(sampler, shuffle=False, drop_last=False)
        dataloader = DataLoader(dataset, 
                                sampler=sampler, 
                                batch_size=batch_size,
                                drop_last=drop_last,
                                worker_init_fn=dataset.collection.torch_worker_init_fn,
                                collate_fn=custom_collate,
                                **dataloader_kwargs)
        print(f'Creating {stage} dataloader by {len(dataloader)} batches of size {batch_size*num_replicas} over {len(dataset)} samples; num_replicas={num_replicas}; sum of indices: {sum(dataset.collection.indices)}')
        return dataloader
        
    def train_dataloader(self):
        dataloader_kwargs = self.dataloader_kwargs['train']
        dataloader = self._get_dataloader(self.train_dataset, dataloader_kwargs, 'train')
        return dataloader

    def val_dataloader(self):
        dataloader_kwargs = self.dataloader_kwargs['val']
        dataloader = self._get_dataloader(self.val_dataset, dataloader_kwargs, 'val')
        return dataloader
