
from numpy.random import choice
from random import shuffle
from torch.utils.data import Sampler
import numpy as np

np.random.seed(42)

        
class WithinGroupSampler(Sampler):
    def __init__(self, obs_list, batch_size, num_samples=None, shuffle=True, drop_last=True):
        self.obs_list = obs_list
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.batches = None
        self._create_batches()
    
    def __len__(self):
        return sum([len(batch) for batch in self.batches])

    def __iter__(self):
        self._create_batches()
        yield from np.hstack(self.batches)

    def _create_batches(self):
        self.batches = []
        count = 0
        for obs in self.obs_list:
            n_obs = len(obs)
            for value in np.unique(obs):
                indices = np.argwhere(obs == value).flatten() + count
                if self.shuffle:
                    indices = choice(indices, len(indices), replace=False)
                num_chunks = int(np.ceil(len(indices) / self.batch_size))
                batches = [indices[i * self.batch_size: (i + 1) * self.batch_size] for i in range(num_chunks)]
                # drop_last
                batches = batches[:-1] if len(batches[-1]) < self.batch_size else batches
                # shuffle(batches)
                self.batches.extend(batches)
            count += n_obs
        shuffle(self.batches)
        if self.num_samples is not None:
            self.batches = self.batches[:self.num_samples//self.batch_size]