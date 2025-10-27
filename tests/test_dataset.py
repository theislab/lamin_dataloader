"""
Tests for the dataset functionality
"""
import pytest
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix
from lamin_dataloader.dataset import TokenizedDataset, GeneIdTokenizer
from lamin_dataloader.utils import normalize
from lamin_dataloader.collections import Collection, InMemoryCollection


@pytest.fixture
def mock_anndata():
    """Fixture that provides a real AnnData instance."""
    # Set random seed for reproducible tests
    np.random.seed(42)
    
    # Create expression data
    n_obs, n_vars = 10, 5
    X = np.random.rand(n_obs, n_vars)
    
    # Create variable names
    var_names = ['gene1', 'gene2', 'gene3', 'gene_not_in_vocab', 'gene4']
    
    # Create variable metadata
    var = pd.DataFrame({
        'gene_symbol': var_names,
        'feature_type': ['Gene Expression'] * n_vars
    })
    
    # Create observation metadata
    obs = pd.DataFrame({
        'cell_type': ['T_cell'] * n_obs,
        'batch': ['batch1'] * n_obs
    })
    
    # Create obsm data
    obsm = {'X_pca': np.random.rand(n_obs, 2)}
    
    # Create AnnData object
    adata = ad.AnnData(X=X, obs=obs, var=var, obsm=obsm)
    adata.var_names = var_names
    
    return adata


class MockCollection(Collection):
    """A mock collection class for testing TokenizedDataset functionality."""
    def __init__(self):
        self.var_list = [['gene1', 'gene2', 'gene_not_in_vocab'], ['gene2', 'gene3', 'gene_not_in_vocab']]
        self._path_list = ['path1', 'path2']
        self.n_obs_list = [10, 5]
        self.join_vars = None  # Mock join_vars as None to use var_list
    
    def __len__(self):
        return sum(self.n_obs_list)
        
    def __getitem__(self, idx):
        if idx < self.n_obs_list[0]:
            return {
                'dataset_id': 0,
                'X': np.array([1.0, 2.0, 3.0])
            }
        else:
            return {
                'dataset_id': 1,
                'X': np.array([4.0, 5.0, 6.0])
            }
    
    @property
    def output_var_list(self):
        """Return the output variable list, mimicking the behavior of MappedCollection."""
        if self.join_vars is not None:
            return [self.var_joint for _ in range(len(self._path_list))]
        else:
            return self.var_list

@pytest.fixture
def mock_collection():
    """Fixture that provides a MockCollection instance."""
    return MockCollection()



def test_inmemory_collection(mock_anndata):
    """Test InMemoryCollection with multiple AnnData objects"""
    # Create two AnnData objects with different data
    adata_1 = mock_anndata[:10].copy()
    adata_2 = mock_anndata[:10].copy()
    adata_2.obs['cell_type'] = 'B_cell'
    adata_2.obs['batch'] = 'batch2'
    adata_list = [adata_1, adata_2]
    n_obs1 = adata_1.n_obs
    n_obs2 = adata_2.n_obs
    var_names1 = adata_1.var_names.values
    var_names2 = adata_2.var_names.values
    
    # Create InMemoryCollection
    collection = InMemoryCollection(
        adata_list=adata_list,
        obs_keys=['cell_type', 'batch'],
        layers_keys=['X'],
        obsm_keys=['X_pca'],
        var_column=None  # Use var_names by default
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
    assert 'X' in item
    assert 'dataset_id' in item
    assert 'cell_type' in item
    assert 'batch' in item
    assert 'obsm_X_pca' in item
    assert item['dataset_id'] == 0
    assert item['cell_type'] == 'T_cell'
    assert item['batch'] == 'batch1'
    assert np.array_equal(item['X'], mock_anndata.X[0])
    
    # Test __getitem__ for second dataset (idx = 10, first obs of second dataset)
    item = collection[n_obs1]
    assert item['dataset_id'] == 1
    assert item['cell_type'] == 'B_cell'
    assert item['batch'] == 'batch2'
    assert np.array_equal(item['X'], mock_anndata.X[0])
    
    # Test __getitem__ for last observation
    item = collection[n_obs1 + n_obs2 - 1]  # Last obs (idx 9 in second dataset)
    assert item['dataset_id'] == 1
    assert np.array_equal(item['X'], mock_anndata.X[9])
    


def test_tokenized_dataset_initialization(mock_collection):
    """Test basic TokenizedDataset initialization"""
    # Create a simple gene mapping for the tokenizer
    gene_mapping = {
        '<pad>': 0,
        '<cls>': 1,
        'gene1': 2,
        'gene2': 3,
        'gene3': 4,
    }
    
    # Create a tokenizer
    tokenizer = GeneIdTokenizer(gene_mapping)
    
    # Create a TokenizedDataset instance
    dataset = TokenizedDataset(
        collection=mock_collection,
        tokenizer=tokenizer,
        obs_keys=[],
        normalization='log1p'
    )
    
    # Basic assertions
    assert dataset is not None
    assert isinstance(dataset, TokenizedDataset)
    
    # Test that the dataset has expected attributes
    assert hasattr(dataset, 'collection')
    assert hasattr(dataset, 'tokenizer')
    assert hasattr(dataset, 'normalization')
    assert hasattr(dataset, 'obs_keys')
    
    # Test dataset length
    assert len(dataset) == 15
    
    # Test getting an item
    item = dataset[0]
    assert 'tokens' in item
    assert 'values' in item
    assert 'dataset_id' in item
    
    # Check if the values returned are correct
    expected_tokens = tokenizer.encode(['gene1', 'gene2'])
    expected_values = normalize(np.array([1.0, 2.0]), 'log1p')
    expected_dataset_id = 0
    print('item', item)
    
    assert np.array_equal(item['tokens'], expected_tokens)
    assert np.allclose(item['values'], expected_values)
    assert item['dataset_id'] == expected_dataset_id


def test_tokenized_dataset_with_inmemory_collection(mock_anndata):
    """Test TokenizedDataset with InMemoryCollection"""
    # Create two AnnData objects with some genes overlapping and some not in vocabulary
    adata_1 = mock_anndata[:10].copy()
    adata_2 = mock_anndata[:10].copy()
    adata_2.obs['cell_type'] = 'B_cell'
    adata_2.obs['batch'] = 'batch2'
    adata_list = [adata_1, adata_2]
    
    # Create InMemoryCollection
    collection = InMemoryCollection(
        adata_list=adata_list,
        obs_keys=['cell_type', 'batch'],
        layers_keys=['X'],
        obsm_keys=['X_pca'],
        var_column=None
    )
    
    # Create a tokenizer with some genes from the mock_anndata vocabulary
    gene_mapping = {
        '<pad>': 0,
        '<cls>': 1,
        'gene1': 2,
        'gene2': 3,
        'gene3': 4,
        'gene4': 5,
    }
    tokenizer = GeneIdTokenizer(gene_mapping)
    
    # Create a TokenizedDataset instance
    dataset = TokenizedDataset(
        collection=collection,
        tokenizer=tokenizer,
        obs_keys=['cell_type', 'batch'],
        obsm_key='X_pca',
        normalization='log1p'
    )
    
    # Basic assertions
    assert dataset is not None
    assert isinstance(dataset, TokenizedDataset)
    
    # Test dataset length (should be total observations from both datasets)
    assert len(dataset) == 20  # 10 + 10
    
    # Test getting an item from the first dataset
    item = dataset[0]
    assert 'tokens' in item
    assert 'values' in item
    assert 'dataset_id' in item
    assert 'cell_type' in item
    assert 'batch' in item
    assert 'X_pca' in item
    
    # Check that tokens exclude 'gene_not_in_vocab'
    assert item['dataset_id'] == 0
    assert len(item['tokens']) == 4  # gene1, gene2, gene3, gene4 (gene_not_in_vocab excluded)
    assert len(item['values']) == 4  # Same as tokens after masking
    assert item['cell_type'] == 'T_cell'
    assert item['batch'] == 'batch1'
    
    # Verify tokens are correct (should be gene1, gene2, gene3, gene4)
    expected_tokens = tokenizer.encode(['gene1', 'gene2', 'gene3', 'gene4'])
    assert np.array_equal(item['tokens'], expected_tokens)
    
    # Verify values are normalized and masked correctly
    expected_values = normalize(mock_anndata.X[0], 'log1p')
    expected_values_masked = expected_values[dataset.masks[0]]  # Using the mask for dataset 0
    assert np.allclose(item['values'], expected_values_masked)
    
    # Test getting an item from the second dataset (index 10)
    item = dataset[10]
    assert item['dataset_id'] == 1
    assert item['cell_type'] == 'B_cell'
    assert item['batch'] == 'batch2'
    assert len(item['tokens']) == 4  # Same genes for second dataset
    assert np.array_equal(item['tokens'], expected_tokens)
    
    # Test masks are computed correctly for both datasets
    assert len(dataset.masks) == 2
    assert len(dataset.tokenized_vars_masked) == 2
    # Both datasets should have the same mask since they have same genes
    assert np.array_equal(dataset.masks[0], dataset.masks[1])


