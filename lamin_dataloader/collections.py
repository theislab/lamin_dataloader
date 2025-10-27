import numpy as np
from numpy.random import choice
from anndata import AnnData
from typing import List

# from lamindb.core import MappedCollection as MappedCollectionMain
from lamin_dataloader._mapped_collection import MappedCollection as MappedCollectionMain

from lamindb.core._mapped_collection import _Connect
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
    def output_var_list(self):
        pass


class InMemoryCollection(Collection):
    """
    A collection class that works with a list of in-memory AnnData objects.
    
    This class provides the same interface as MappedCollection but works with
    AnnData objects that are already loaded in memory.
    """
    
    def __init__(self, adata_list: List[AnnData], obs_keys=[], layers_keys=['X'], obsm_keys=None, var_column=None):
        """
        Initialize the InMemoryCollection.
        
        Args:
            adata_list: List of AnnData objects containing the data
            obs_keys: List of keys from adata.obs to include in the output
            layers_keys: List of layer keys to extract (default: ['X'])
            obsm_keys: List of obsm keys to extract (default: None)
            var_column: Column name in adata.var to use as variable names (default: None, uses index)
        """
        self.adata_list = adata_list
        self.obs_keys = obs_keys
        self.layers_keys = layers_keys
        self.obsm_keys = obsm_keys
        self.var_column = var_column
        
        # Compute n_obs_list for each storage
        self.n_obs_list = [ad.n_obs for ad in self.adata_list]
        self.n_obs = sum(self.n_obs_list)
        
        # Create indices and storage_idx arrays
        # indices: concatenated array of indices for each storage (0, 1, 2, ..., n_obs-1 for each storage)
        # storage_idx: which storage each sample belongs to
        self.indices = np.hstack([np.arange(n_obs) for n_obs in self.n_obs_list])
        self.storage_idx = np.repeat(np.arange(len(self.adata_list)), self.n_obs_list)
        
        # Get variable names for each storage
        self.var_names_list = []
        for i, ad in enumerate(self.adata_list):
            if var_column is not None:
                if var_column in ad.var.columns:
                    var_names = ad.var[var_column].values
                else:
                    raise ValueError(f"Column '{var_column}' not found in adata[{i}].var")
            else:
                var_names = ad.var_names.values
            self.var_names_list.append(var_names)
        
        self._path_list = ['in_memory'] * len(self.adata_list)
        
    def __len__(self):
        return self.n_obs
    
    def __getitem__(self, idx):
        """
        Get a sample from the collection.
        
        Returns a dict with 'X', 'dataset_id', and any requested obs_keys and obsm_keys.
        """
        # Get the observation index and storage index
        obs_idx = self.indices[idx]
        storage_idx = self.storage_idx[idx]
        adata = self.adata_list[storage_idx]
        
        out = {}
        
        # Extract layer data (e.g., X)
        for layers_key in self.layers_keys:
            if layers_key == "X":
                # Handle sparse matrices
                X = adata.X[obs_idx]
                if hasattr(X, 'toarray'):
                    X = X.toarray().flatten()
                out['X'] = X
            else:
                if layers_key in adata.layers:
                    layer_data = adata.layers[layers_key][obs_idx]
                    if hasattr(layer_data, 'toarray'):
                        layer_data = layer_data.toarray().flatten()
                    out[layers_key] = layer_data
        
        # Extract obsm data
        if self.obsm_keys is not None:
            for obsm_key in self.obsm_keys:
                if obsm_key in adata.obsm:
                    out[f"obsm_{obsm_key}"] = adata.obsm[obsm_key][obs_idx]
        
        # Extract obs data
        for key in self.obs_keys:
            if key in adata.obs.columns:
                out[key] = adata.obs[key].iloc[obs_idx]
        
        # Set dataset_id to the storage index
        out['dataset_id'] = storage_idx
        
        return out
    
    @property
    def output_var_list(self):
        """
        Return the variable list as a list containing arrays for each dataset.
        This matches the interface expected by TokenizedDataset.
        """
        return self.var_names_list


class MappedCollection(MappedCollectionMain, Collection):

    def __init__(self, *args, **kwargs):
        
        self.keys_to_cache = None
        if 'keys_to_cache' in kwargs:
            self.keys_to_cache = kwargs.pop('keys_to_cache')
            
        super().__init__(*args, **kwargs)
        self._validate_data()
        self.var_column = None
        
        # _cached_obs: {key: [np.array of values for each storage]}
        self._cached_obs = {}
        if self.keys_to_cache is not None:
            for key in self.keys_to_cache:
                print(f'Caching {key}...')
                self._cache_key(key)
        
        
        self._make_join_vars()
        
    
    def _validate_data(self):
        for storage in self.storages:
            with _Connect(storage) as store:
                layer = store["raw"]["X"] if 'raw' in store.keys() else store["X"]
                # check if the sparse matrix is csr_matrix:
                assert dict(layer.attrs)['encoding-type'] != 'csc_matrix', f'Only csr_matrix is supported for sparse arrays. storage {storage} is not csr_matrix.'
                
                # check if it's really raw data: 
                # assert (layer["data"][:10] == np.round(layer["data"][:10])).all(), f'storage {storage} is not raw data.'
                
                for col in self.obs_keys:
                    assert col in store["obs"].keys(), f'{col} is not in obs keys of storage {storage}'
                
                if self.keys_to_cache is not None:
                    for key in self.keys_to_cache:
                        if key != 'dataset' and key is not None:
                            assert key in store["obs"].keys(), f'{key} is not in obs keys of storage {storage}'
        
        
    @property
    def output_var_list(self):
        if self.join_vars is not None:
            print(f'var_joint: {self.var_joint}')
            return [self.var_joint for _ in range(len(self.storages))]
        
        else:
            if self.var_list is None:
                self._read_vars()
            return self.var_list


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
            cats = sorted(cats) # Added: This is to keep the mapping consistent across differnt runs
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
                    if obsm_key in store["obsm"].keys():
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
    


    def _cache_key(self, key: str):
        if key == 'dataset':
            self._cached_obs['dataset'] = [np.repeat(i, n) for i, n in enumerate(self.n_obs_list)]
        elif key is not None:
            self._cached_obs[key] = []
            for i, storage in enumerate(self.storages):
                with _Connect(storage) as store:
                    values = self._get_labels(store, key, storage_idx=i)
                    self._cached_obs[key].append(np.array(values))
        
                    

    # def _get_obs_values(self, storage: StorageType, label_key: str):
    #     """Get categories."""
    #     obs = storage["obs"]  # type: ignore
    #     labels = obs[label_key]
    #     assert isinstance(labels, GroupTypes), "Only GroupTypes are supported."
    #     if "codes" in labels:
    #         cats = self._get_categories(storage, label_key)
    #         labels = [cats[code] for code in labels["codes"]]
    #         if isinstance(labels[0], bytes):
    #             labels = [l.decode("utf-8") for l in labels]
    #         return labels
    #     else:
    #         raise ValueError(f"Group {label_key} is not categorical.")
    
    
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
        mapped._make_connections(mapped.path_list, parallel=False)