import numpy as np
from numpy.random import choice

from lamindb.core import MappedCollection
from lamindb.core._mapped_collection import _Connect
from lamindb.core.storage._anndata_accessor import _safer_read_index


class CustomMappedCollection(MappedCollection):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._validate_data()
        self._make_var_list(var_column=None) # var_column does not work for now
        
    
    def _validate_data(self):
        for storage in self.storages:
            with _Connect(storage) as store:
                layer = store["raw"]["X"] if 'raw' in store.keys() else store["X"]
                # check if the sparse matrix is csr_matrix:
                assert dict(layer.attrs)['encoding-type'] != 'csc_matrix', f'Only csr_matrix is supported for sparse arrays. storage {storage} is not csr_matrix.'
                
                # check if it's really raw data: 
                assert (layer["data"][:10] == np.round(layer["data"][:10])).all(), f'storage {storage} is not raw data.'
        
        
    def _make_var_list(self, var_column=None):
        var_list = []
        for storage in self.storages:
            with _Connect(storage) as store:
                if var_column is not None:
                    array = store['var'][var_column]
                    if dict(store['var'][var_column].attrs)['encoding-type'] == 'categorical':
                        array = array['categories'][array['codes']]
                    var_list.append(np.array([x.decode("utf-8") for x in array]))
                else:
                    var_list.append(_safer_read_index(store["var"]))
        self.var_list = var_list


    def subset_data(self, sub_sample_frac):
        self.n_obs_list_orig = self.n_obs_list
        self.n_obs_list = [int(n_obs * sub_sample_frac) for n_obs in self.n_obs_list]
        self.n_obs = sum(self.n_obs_list)
        self.indices = np.hstack([choice(np.arange(n_obs_orig), n_obs, replace=False) for n_obs, n_obs_orig in zip(self.n_obs_list, self.n_obs_list_orig)])
        self.storage_idx = np.repeat(np.arange(len(self.storages)), self.n_obs_list)


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
        return out
    
    
    @staticmethod
    def torch_worker_init_fn(worker_id):
        """`worker_init_fn` for `torch.utils.data.DataLoader`.

        Improves performance for `num_workers > 1`.
        """
        from torch.utils.data import get_worker_info

        mapped = get_worker_info().dataset.mc
        mapped.parallel = False
        mapped.storages = []
        mapped.conns = []
        mapped._make_connections(mapped._path_list, parallel=False)