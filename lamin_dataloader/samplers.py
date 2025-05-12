
from numpy.random import choice
from random import shuffle
from torch.utils.data import Sampler
import numpy as np

np.random.seed(42)


# Use this sampler to only take the samples are in obs_filter
class SubsetSampler(Sampler):
    def __init__(self, storage_idx, obs_list_dict, obs_filter_dict, batch_size, num_samples=None, shuffle=True, drop_last=True, stage='train'):
        self.storage_idx = storage_idx
        self.obs_list_dict = obs_list_dict
        self.obs_filter_dict = obs_filter_dict
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.batches = None
        self._create_batches()
        self.stage = stage
        assert stage in ['train', 'val', 'test'], 'stage must be one of "train", "val", "test"'
    
    def __len__(self):
        return sum([len(batch) for batch in self.batches])

    def __iter__(self):
        if self.stage == 'train':
            self._create_batches()
        yield from np.hstack(self.batches)

    def _create_batches(self):
        self.batches = []
        
        indices = []
        filters = self.obs_filter_dict.keys()
        for filter_key in filters:
            obs_list = self.obs_list_dict[filter_key]
            obs_filter = self.obs_filter_dict[filter_key]
            obs = np.concatenate(obs_list)
            idx = np.argwhere(np.isin(obs, obs_filter)).squeeze()
            indices.append(idx)
        indices = list(set.intersection(*map(set, indices)))
        if len(indices) == 0:
            raise ValueError('No samples found in the given filters.')
        
        if self.shuffle:
            indices = choice(indices, len(indices), replace=False)
        num_chunks = int(np.ceil(len(indices) / self.batch_size))
        batches = [indices[i * self.batch_size: (i + 1) * self.batch_size] for i in range(num_chunks)]
        # drop_last
        self.batches = batches[:-1] if len(batches[-1]) < self.batch_size else batches
        
        if self.num_samples is not None:
            self.batches = self.batches[:self.num_samples//self.batch_size]