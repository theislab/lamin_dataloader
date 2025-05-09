import os
from typing import Dict, List
import lightning as L
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler
from lamin_dataloader.dataset import TokenizedDataset, Tokenizer
from lamin_dataloader.dataset import CustomCollate
from lamin_dataloader.collections import MappedCollection
from lightning.fabric.utilities.distributed import DistributedSamplerWrapper
import multiprocessing

class MappedCollectionDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_path: str,
        split: Dict,
        tokenizer: Tokenizer,
        columns: List[str],
        normalization: str = 'log1p',
        gene_sampling_strategy: str = 'random',
        model_speed_sanity_check: bool = False,
        dataset_kwargs: Dict = {},
        dataloader_kwargs: Dict = {},
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.gene_sampling_strategy = gene_sampling_strategy
        self.model_speed_sanity_check = model_speed_sanity_check
        self.dataloader_kwargs = dataloader_kwargs

        dataset_kwargs_shared = {'obs_keys': columns, 
                                 'tokenizer': tokenizer, 
                                 'normalization': normalization
                                 }


        if 'train' in split and split['train'] is not None and 'train' in dataset_kwargs:
            path_list = [os.path.join(dataset_path, file) for file in split['train']]
            self.train_collate_fn = self._get_collate_fn(dataset_kwargs['train'])
            collection = MappedCollection(path_list, layers_keys="X", obs_keys=columns,join=None, encode_labels=True, parallel=True)
            self.train_dataset = TokenizedDataset(**{'collection': collection, **dataset_kwargs_shared, **dataset_kwargs['train']})
        
        if 'val' in split and split['val'] is not None and 'val' in dataset_kwargs:
            path_list = [os.path.join(dataset_path, file) for file in split['val']]
            self.val_collate_fn = self._get_collate_fn(dataset_kwargs['val'])
            collection = MappedCollection(path_list, layers_keys="X", obs_keys=columns, join=None, encode_labels=True, parallel=True)
            self.val_dataset = TokenizedDataset(**{'collection': collection, **dataset_kwargs_shared, **dataset_kwargs['val']})
            
        if 'test' in split and split['test'] is not None and 'test' in dataset_kwargs:
            path_list = [os.path.join(dataset_path, file) for file in split['test']]
            self.test_collate_fn = self._get_collate_fn(dataset_kwargs['test'], split_input=False)
            collection = MappedCollection(path_list, layers_keys="X", obs_keys=columns, join=None, encode_labels=True, parallel=True)
            self.test_dataset = TokenizedDataset(**{'collection': collection, **dataset_kwargs_shared, **dataset_kwargs['test']})

        self._val_dataloader = None
    
    def _get_collate_fn(self, dataset_kwargs):
        return CustomCollate(
            tokenizer=self.tokenizer,
            max_tokens=dataset_kwargs.pop('max_tokens'),
            )
    
    def _get_dataloader(self, dataset, dataloader_kwargs, collate_fn, stage):
        num_replicas = dist.get_world_size() if torch.distributed.is_initialized() else 1
        batch_size = dataloader_kwargs.pop('batch_size') // num_replicas
        shuffle = dataloader_kwargs.pop('shuffle')
        drop_last = dataloader_kwargs.pop('drop_last')
        num_samples = dataloader_kwargs.pop('num_samples')
        num_workers = dataloader_kwargs.pop('num_workers')
        num_workers = min(int(os.getenv("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count())), num_workers)
        
        assert drop_last == True, 'drop_last must be True during training and validation'
        assert shuffle == True, 'shuffle must be True during training and validation'
        
        if num_samples is not None and num_samples >= len(dataset):
            print(f'Warning: num_samples ({num_samples}) is greater than or equal to the number of samples in the dataset ({len(dataset)}).')

        sampler = RandomSampler(dataset, num_samples=num_samples)
        
        if torch.distributed.is_initialized():
            sampler = DistributedSamplerWrapper(sampler, shuffle=False, drop_last=False)

        dataloader = DataLoader(dataset, 
                                sampler=sampler, 
                                batch_size=batch_size,
                                drop_last=drop_last,
                                worker_init_fn=dataset.collection.torch_worker_init_fn,
                                collate_fn=collate_fn,
                                num_workers=num_workers,
                                persistent_workers=True,
                                **dataloader_kwargs)
        print(f'Creating {stage} dataloader by {len(dataloader)} batches of size {batch_size*num_replicas} taking {len(dataloader)*batch_size*num_replicas} samples from {len(dataset)} total samples; num_replicas={num_replicas}; sum of indices: {sum(dataset.collection.indices)}; num_workers={num_workers}')
        return dataloader
        
    def train_dataloader(self):
        dataloader_kwargs = self.dataloader_kwargs['train']
        dataloader = self._get_dataloader(self.train_dataset, dataloader_kwargs, self.train_collate_fn, 'train')
        return dataloader

    def val_dataloader(self):
        dataloader_kwargs = self.dataloader_kwargs['val']
        dataloader = self._get_dataloader(self.val_dataset, dataloader_kwargs, self.val_collate_fn, 'train')
        return dataloader
        
    def test_dataloader(self):
        dataloader_kwargs = self.dataloader_kwargs['test']
        
        assert dataloader_kwargs['shuffle'] == False, 'shuffle should be false for test dataloader'
        assert dataloader_kwargs['drop_last'] == False, 'drop_last should be false for test dataloader'
        dataloader = DataLoader(self.test_dataset, 
                                worker_init_fn=self.test_dataset.collection.torch_worker_init_fn, 
                                collate_fn=self.test_collate_fn, 
                                **dataloader_kwargs)
        print(f'Creating test dataloader by {len(dataloader)} batches of size {dataloader_kwargs["batch_size"]} over {len(self.test_dataset)} samples; sum of indices: {sum(self.test_dataset.collection.indices)}')
        return dataloader