import numpy as np
from numpy.random import choice

from lamindb.core import MappedCollection as MappedCollectionMain
from lamindb.core._mapped_collection import _Connect
from lamindb.core.storage._anndata_accessor import _safer_read_index
from abc import ABC, abstractmethod

from lamindb.core.storage._anndata_accessor import (
    ArrayType,
    ArrayTypes,
    GroupType,
    GroupTypes,
    StorageType,
    _safer_read_index,
    get_spec,
    registry,
)

class Collection(ABC):
    """
    An abstract class for mapped collections.
    """
    
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def __len__(self):
        pass

    # The __getitem__ method must return a Dict containing the dataset_id of 
    # the sample and the count data in the "X" key and any other obs keys with 
    # their respective keys like {"X": count_data, "dataset_id": dataset_id, 'cell_type': cell_type}
    @abstractmethod
    def __getitem__(self, idx):
        pass

    
    @property
    @abstractmethod
    def var_list(self):
        pass


class MappedCollection(MappedCollectionMain, Collection):

    def __init__(self, *args, **kwargs):
        
        self.keys_to_cache = None
        if 'keys_to_cache' in kwargs:
            self.keys_to_cache = kwargs.pop('keys_to_cache')
            
        super().__init__(*args, **kwargs)
        self._validate_data()
        self.var_column = None
        
        self._cached_obs = {}
        if self.keys_to_cache is not None:
            for key in self.keys_to_cache:
                print(f'Caching {key}...')
                self._cache_key(key)
        
        
    
    def _validate_data(self):
        for storage in self.storages:
            with _Connect(storage) as store:
                layer = store["raw"]["X"] if 'raw' in store.keys() else store["X"]
                # check if the sparse matrix is csr_matrix:
                assert dict(layer.attrs)['encoding-type'] != 'csc_matrix', f'Only csr_matrix is supported for sparse arrays. storage {storage} is not csr_matrix.'
                
                # check if it's really raw data: 
                assert (layer["data"][:10] == np.round(layer["data"][:10])).all(), f'storage {storage} is not raw data.'
                
                for col in self.obs_keys:
                    assert col in store["obs"].keys(), f'{col} is not in obs keys of storage {storage}'
                
                if self.keys_to_cache is not None:
                    for key in self.keys_to_cache:
                        if key != 'dataset' and key is not None:
                            assert key in store["obs"].keys(), f'{key} is not in obs keys of storage {storage}'
        
        
    @property
    def var_list(self):
        if self.var_joint is not None:
            print(f'var_joint: {self.var_joint}')
            return [self.var_joint for _ in range(len(self.storages))]
        
        if not hasattr(self, "_var_list"):
            self._var_list = []
            for storage in self.storages:
                with _Connect(storage) as store:
                    if self.var_column is not None:
                        array = store['var'][self.var_column]
                        if dict(store['var'][self.var_column].attrs)['encoding-type'] == 'categorical':
                            array = array['categories'][array['codes']]
                        self._var_list.append(np.array([x.decode("utf-8") for x in array]))
                    else:
                        self._var_list.append(_safer_read_index(store["var"]))
        return self._var_list


    # def subset_data(self, sub_sample_frac):
    #     self.n_obs_list_orig = self.n_obs_list
    #     self.n_obs_list = [int(n_obs * sub_sample_frac) for n_obs in self.n_obs_list]
    #     self.n_obs = sum(self.n_obs_list)
    #     self.indices = np.hstack([choice(np.arange(n_obs_orig), n_obs, replace=False) for n_obs, n_obs_orig in zip(self.n_obs_list, self.n_obs_list_orig)])
    #     self.storage_idx = np.repeat(np.arange(len(self.storages)), self.n_obs_list)


    def _make_encoders(self, encode_labels: list):
        for label in encode_labels:
            cats = self.get_merged_categories(label)
            encoder = {}
            if isinstance(self.unknown_label, dict):
                unknown_label = self.unknown_label.get(label, None)
            else:
                unknown_label = self.unknown_label
            if unknown_label is not None and unknown_label in cats:
                cats.remove(unknown_label)
                encoder[unknown_label] = -1
            cats = sorted(cats) # added
            encoder.update({cat: i for i, cat in enumerate(cats)})
            self.encoders[label] = encoder


    def __getitem__(self, idx: int):
        obs_idx = self.indices[idx]
        storage_idx = self.storage_idx[idx]
        if self.var_indices is not None:
            var_idxs_join = self.var_indices[storage_idx]
        else:
            var_idxs_join = None

        with _Connect(self.storages[storage_idx]) as store:
            out = {}
            for layers_key in self.layers_keys:
                # lazy_data = (
                #     store["X"] if layers_key == "X" else store["layers"][layers_key]
                # )
                # added:
                if layers_key == "X":
                    lazy_data = store["raw"]["X"] if 'raw' in store.keys() else store["X"]
                else:
                    lazy_data = store["layers"][layers_key]
                    
                out[layers_key] = self._get_data_idx(
                    lazy_data, obs_idx, self.join_vars, var_idxs_join, self.n_vars
                )
            if self.obsm_keys is not None:
                for obsm_key in self.obsm_keys:
                    lazy_data = store["obsm"][obsm_key]
                    out[f"obsm_{obsm_key}"] = self._get_data_idx(lazy_data, obs_idx)
            out["_store_idx"] = storage_idx
            if self.obs_keys is not None:
                for label in self.obs_keys:
                    if label in self._cache_cats:
                        cats = self._cache_cats[label][storage_idx]
                        if cats is None:
                            cats = []
                    else:
                        cats = None
                    label_idx = self._get_obs_idx(store, obs_idx, label, cats)
                    if label in self.encoders:
                        label_idx = self.encoders[label][label_idx]
                    out[label] = label_idx
            out['dataset_id'] = out["_store_idx"]
        return out
    


    def _cache_key(self, key: list):
        if key == 'dataset':
            self._cached_obs['dataset'] = [np.repeat(i, n) for i, n in enumerate(self.n_obs_list)]
        elif key is not None:
            self._cached_obs[key] = []
            for storage in self.storages:
                with _Connect(storage) as store:
                    values = self._get_obs_values(store, key)
                    self._cached_obs[key].append(np.array(values))
        
                    

    def _get_obs_values(self, storage: StorageType, label_key: str):
        """Get categories."""
        obs = storage["obs"]  # type: ignore
        labels = obs[label_key]
        assert isinstance(labels, GroupTypes), "Only GroupTypes are supported."
        if "codes" in labels:
            cats = self._get_categories(storage, label_key)
            labels = [cats[code] for code in labels["codes"]]
            if isinstance(labels[0], bytes):
                labels = [l.decode("utf-8") for l in labels]
            return labels
        else:
            raise ValueError(f"Group {label_key} is not categorical.")
    
    
    @staticmethod
    def torch_worker_init_fn(worker_id):
        """`worker_init_fn` for `torch.utils.data.DataLoader`.

        Improves performance for `num_workers > 1`.
        """
        from torch.utils.data import get_worker_info

        mapped = get_worker_info().dataset.collection
        mapped.parallel = False
        mapped.storages = []
        mapped.conns = []
        mapped._make_connections(mapped._path_list, parallel=False)