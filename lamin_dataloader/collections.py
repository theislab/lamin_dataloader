from abc import ABC, abstractmethod
import numpy as np
from numpy.random import choice
from anndata import AnnData
from typing import List


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
    
    def __init__(self, adata_list: List[AnnData], obs_keys=[], layers_keys=['X'], obsm_keys=None, var_column=None, keys_to_cache=None):
        """
        Initialize the InMemoryCollection.
        
        Args:
            adata_list: List of AnnData objects containing the data
            obs_keys: List of keys from adata.obs to include in the output
            layers_keys: List of layer keys to extract (default: ['X'])
            obsm_keys: List of obsm keys to extract (default: None)
            var_column: Column name in adata.var to use as variable names (default: None, uses index)
            keys_to_cache: List of obs keys appended for faster access (default: None)
        """
        self.adata_list = adata_list
        self.obs_keys = obs_keys
        self.layers_keys = layers_keys
        self.obsm_keys = obsm_keys
        self.var_column = var_column
        self.keys_to_cache = keys_to_cache
        
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
        
        # _cached_obs: {key: [np.array of values for each storage]}
        self._cached_obs = {}
        if self.keys_to_cache is not None:
            for key in self.keys_to_cache:
                print(f'Caching {key}...')
                self._cache_key(key)
        
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
    
    def _cache_key(self, key: str):
        """Cache an obs key for faster access."""
        if key == 'dataset':
            self._cached_obs['dataset'] = [np.repeat(i, n) for i, n in enumerate(self.n_obs_list)]
        elif key is not None:
            self._cached_obs[key] = []
            for adata in self.adata_list:
                values = adata.obs[key].values
                self._cached_obs[key].append(np.array(values))
