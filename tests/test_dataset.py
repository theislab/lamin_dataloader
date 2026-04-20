"""
Tests for the dataset functionality
"""

import numpy as np
from lamin_dataloader.dataset import TokenizedDataset
from lamin_dataloader.utils import normalize
from lamin_dataloader.collections import InMemoryCollection


def test_inmemory_collection(adata):
    """Test InMemoryCollection with multiple AnnData objects"""
    # Create two AnnData objects with different data
    adata_1 = adata[:15].copy()
    adata_1.obs["cell_type"] = "T_cell"
    adata_1.obs["batch"] = "batch1"
    adata_1.obsm["X_pca"] = np.random.rand(adata_1.n_obs, 2)

    adata_2 = adata[15:].copy()
    adata_2.obs["cell_type"] = "B_cell"
    adata_2.obs["batch"] = "batch2"
    adata_2.obsm["X_pca"] = np.random.rand(adata_2.n_obs, 2)

    adata_list = [adata_1, adata_2]
    n_obs1 = adata_1.n_obs
    n_obs2 = adata_2.n_obs
    var_names1 = adata_1.var_names.values
    var_names2 = adata_2.var_names.values

    # Create InMemoryCollection
    collection = InMemoryCollection(
        adata_list=adata_list,
        obs_keys=["cell_type", "batch"],
        layers_keys=["X"],
        obsm_keys=["X_pca"],
        var_column=None,  # Use var_names by default
    )

    # Test initialization
    assert len(collection) == n_obs1 + n_obs2
    assert len(collection.n_obs_list) == 2
    assert collection.n_obs_list == [n_obs1, n_obs2]

    # Test output_var_list
    assert len(collection.output_var_list) == 2
    assert np.array_equal(collection.output_var_list[0], np.asarray(var_names1))
    assert np.array_equal(collection.output_var_list[1], np.asarray(var_names2))

    # Test __getitem__ for first dataset
    item = collection[0]
    assert "X" in item
    assert "dataset" in item
    assert "cell_type" in item
    assert "batch" in item
    assert "obsm_X_pca" in item
    assert item["dataset"] == 0
    assert item["cell_type"] == "T_cell"
    assert item["batch"] == "batch1"
    assert np.array_equal(item["X"], adata_1.X[0])

    # Test __getitem__ for second dataset (idx = n_obs1, first obs of second dataset)
    item = collection[n_obs1]
    assert item["dataset"] == 1
    assert item["cell_type"] == "B_cell"
    assert item["batch"] == "batch2"
    assert np.array_equal(item["X"], adata_2.X[0])

    # Test __getitem__ for last observation
    item = collection[n_obs1 + n_obs2 - 1]  # Last obs (last obs in second dataset)
    assert item["dataset"] == 1
    assert np.array_equal(item["X"], adata_2.X[-1])


def test_tokenized_dataset_with_inmemory_collection(adata, tokenizer):
    """Test TokenizedDataset with InMemoryCollection"""
    # Create two AnnData objects with the same genes
    adata_1 = adata[:10].copy()
    adata_1.obs["cell_type"] = "T_cell"
    adata_1.obs["batch"] = "batch1"
    adata_1.obsm["X_pca"] = np.random.rand(adata_1.n_obs, 2)

    adata_2 = adata[10:20].copy()
    adata_2.obs["cell_type"] = "B_cell"
    adata_2.obs["batch"] = "batch2"
    adata_2.obsm["X_pca"] = np.random.rand(adata_2.n_obs, 2)

    adata_list = [adata_1, adata_2]

    # Create InMemoryCollection
    collection = InMemoryCollection(
        adata_list=adata_list, obs_keys=["cell_type", "batch"], layers_keys=["X"], obsm_keys=["X_pca"], var_column=None
    )

    # Create a TokenizedDataset instance
    dataset = TokenizedDataset(
        collection=collection,
        tokenizer=tokenizer,
        obs_keys=["cell_type", "batch"],
        obsm_key="X_pca",
        normalization="log1p",
    )

    # Basic assertions
    assert dataset is not None
    assert isinstance(dataset, TokenizedDataset)

    # Test dataset length (should be total observations from both datasets)
    assert len(dataset) == 20  # 10 + 10

    # Test getting an item from the first dataset
    item = dataset[0]
    assert "tokens" in item
    assert "values" in item
    assert "dataset" in item
    assert "cell_type" in item
    assert "batch" in item
    assert "X_pca" in item

    # Check basic properties
    assert item["dataset"] == 0
    assert len(item["tokens"]) == len(adata_1.var_names)  # All genes should be tokenized
    assert len(item["values"]) == len(item["tokens"])  # Same length after masking
    assert item["cell_type"] == "T_cell"
    assert item["batch"] == "batch1"

    # Verify tokens are correctly encoded
    expected_tokens = tokenizer.encode(adata_1.var_names.tolist())
    assert np.array_equal(item["tokens"], expected_tokens)

    # Verify values are normalized and masked correctly
    expected_values = normalize(adata_1.X[0], "log1p")
    expected_values_masked = expected_values[dataset.masks[0]]  # Using the mask for dataset 0
    assert np.allclose(item["values"], expected_values_masked)

    # Test getting an item from the second dataset (index 10)
    item = dataset[10]
    assert item["dataset"] == 1
    assert item["cell_type"] == "B_cell"
    assert item["batch"] == "batch2"
    assert len(item["tokens"]) == len(adata_2.var_names)  # Same genes for second dataset
    assert np.array_equal(item["tokens"], expected_tokens)

    # Test masks are computed correctly for both datasets
    assert len(dataset.masks) == 2
    assert len(dataset.tokenized_vars_masked) == 2
    # Both datasets should have the same mask since they have same genes
    assert np.array_equal(dataset.masks[0], dataset.masks[1])
