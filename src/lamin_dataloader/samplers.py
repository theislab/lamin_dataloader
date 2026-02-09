import os

import numpy as np
from torch.utils.data import Sampler


# Use this sampler to only take the samples are in obs_filter
class SubsetSampler(Sampler):
    def __init__(
        self,
        dataset_size,
        batch_size,
        obs_list_dict,
        obs_filter_dict=None,
        num_samples=None,
        shuffle=True,
        drop_last=True,
        stage="train",
        start_epoch=0,
    ):
        self.dataset_size = dataset_size
        self.obs_list_dict = obs_list_dict
        self.obs_filter_dict = obs_filter_dict
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.batches = None
        self.stage = stage
        self.current_epoch = start_epoch
        assert stage in ["train", "val", "test"], 'stage must be one of "train", "val", "test"'
        self.seed = int(os.environ.get("PL_GLOBAL_SEED", 42))
        self._create_batches()

    def __len__(self):
        return sum([len(batch) for batch in self.batches])

    def __iter__(self):
        yield from np.hstack(self.batches)
        if self.stage == "train":
            self._create_batches()

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def _create_batches(self):
        # Create RNG instance based on current epoch to ensure reproducibility
        if self.stage == "train":
            rng = np.random.default_rng(self.seed * 10_000 + self.current_epoch)
        else:
            rng = np.random.default_rng(self.seed)  # for validation and test

        self.batches = []

        if self.obs_filter_dict is None or len(self.obs_filter_dict) == 0:
            indices = list(range(self.dataset_size))
        else:
            indices = []
            filters = self.obs_filter_dict.keys()
            for filter_key in filters:
                obs = self.obs_list_dict[filter_key]
                obs_filter = self.obs_filter_dict[filter_key]
                idx = np.argwhere(np.isin(obs, obs_filter)).squeeze()
                indices.append(idx)
            indices = list(set.intersection(*map(set, indices)))
            if len(indices) == 0:
                raise ValueError("No samples found in the given filters.")

        if self.shuffle:
            indices = rng.choice(indices, len(indices), replace=False)
        num_chunks = int(np.ceil(len(indices) / self.batch_size))
        batches = [indices[i * self.batch_size : (i + 1) * self.batch_size] for i in range(num_chunks)]
        # drop_last
        self.batches = batches[:-1] if len(batches[-1]) < self.batch_size else batches

        if self.num_samples is not None:
            self.batches = self.batches[: self.num_samples // self.batch_size]
