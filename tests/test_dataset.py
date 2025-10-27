"""
Tests for the dataset functionality
"""
import pytest
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix
from lamin_dataloader.dataset import TokenizedDataset, InMemoryTokenizedDataset, GeneIdTokenizer
from lamin_dataloader.utils import normalize
from lamin_dataloader.collections import Collection

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


def test_inmemory_tokenized_dataset_initialization(mock_anndata):
    """Test basic InMemoryTokenizedDataset initialization"""
    # Create a simple gene mapping for the tokenizer
    gene_mapping = {
        '<pad>': 0,
        '<cls>': 1,
        'gene1': 2,
        'gene2': 3,
        'gene3': 4,
        'gene4': 5,
    }
    
    # Create a tokenizer
    tokenizer = GeneIdTokenizer(gene_mapping)
    
    # Create an InMemoryTokenizedDataset instance
    dataset = InMemoryTokenizedDataset(
        adata=mock_anndata,
        tokenizer=tokenizer,
        obs_keys=['cell_type', 'batch'],
        obsm_key='X_pca',
        normalization='log1p'
    )
    
    # Basic assertions
    assert dataset is not None
    assert isinstance(dataset, InMemoryTokenizedDataset)
    
    # Test that the dataset has expected attributes
    assert hasattr(dataset, 'adata')
    assert hasattr(dataset, 'tokenizer')
    assert hasattr(dataset, 'normalization')
    assert hasattr(dataset, 'obs_keys')
    assert hasattr(dataset, 'obsm_key')
    assert hasattr(dataset, 'var_names')
    assert hasattr(dataset, 'tokenized_vars')
    assert hasattr(dataset, 'mask')
    assert hasattr(dataset, 'tokenized_vars_masked')
    
    # Test dataset length
    assert len(dataset) == 10
    
    # Test that mask is correctly applied (gene_not_in_vocab should be masked out)
    expected_mask = np.array([True, True, True, False, True])  # gene_not_in_vocab should be False
    assert np.array_equal(dataset.mask, expected_mask)
    
    # Test that tokenized_vars_masked only contains valid tokens
    assert len(dataset.tokenized_vars_masked) == 4  # 4 valid genes
    assert all(token != tokenizer.NOT_FOUND for token in dataset.tokenized_vars_masked)


def test_inmemory_tokenized_dataset_getitem(mock_anndata):
    """Test InMemoryTokenizedDataset __getitem__ method"""
    # Create a simple gene mapping for the tokenizer
    gene_mapping = {
        '<pad>': 0,
        '<cls>': 1,
        'gene1': 2,
        'gene2': 3,
        'gene3': 4,
        'gene4': 5,
    }
    
    # Create a tokenizer
    tokenizer = GeneIdTokenizer(gene_mapping)
    
    # Create an InMemoryTokenizedDataset instance
    dataset = InMemoryTokenizedDataset(
        adata=mock_anndata,
        tokenizer=tokenizer,
        obs_keys=['cell_type', 'batch'],
        obsm_key='X_pca',
        normalization='log1p'
    )
    
    # Test getting an item
    item = dataset[0]
    
    # Check required keys are present
    assert 'tokens' in item
    assert 'values' in item
    assert 'dataset_id' in item
    assert 'cell_type' in item
    assert 'batch' in item
    assert 'X_pca' in item
    
    # Check data types and shapes
    assert isinstance(item['tokens'], np.ndarray)
    assert isinstance(item['values'], np.ndarray)
    assert item['dataset_id'] == 0  # Always 0 for single dataset
    assert len(item['tokens']) == 4  # 4 valid genes after masking
    assert len(item['values']) == 4  # Same length as tokens
    
    # Check that values are normalized
    expected_values = normalize(mock_anndata.X[0], 'log1p')
    expected_values_masked = expected_values[dataset.mask]
    assert np.allclose(item['values'], expected_values_masked)
    
    # Check that tokens are correct
    expected_tokens = tokenizer.encode(['gene1', 'gene2', 'gene3', 'gene4'])
    assert np.array_equal(item['tokens'], expected_tokens)
    
    # Check obs data
    assert item['cell_type'] == 'T_cell'
    assert item['batch'] == 'batch1'
    
    # Check obsm data
    assert np.array_equal(item['X_pca'], mock_anndata.obsm['X_pca'][0])



def test_inmemory_tokenized_dataset_no_valid_tokens():
    """Test InMemoryTokenizedDataset when no tokens are found in vocabulary"""
    # Create AnnData with genes not in vocabulary
    n_obs, n_vars = 5, 3
    X = np.random.rand(n_obs, n_vars)
    var_names = ['unknown_gene1', 'unknown_gene2', 'unknown_gene3']
    
    adata = ad.AnnData(X=X)
    adata.var_names = var_names
    
    # Create a tokenizer with different genes
    gene_mapping = {
        '<pad>': 0,
        '<cls>': 1,
        'gene1': 2,
        'gene2': 3,
    }
    tokenizer = GeneIdTokenizer(gene_mapping)
    
    # Test that AssertionError is raised when no tokens are found
    with pytest.raises(AssertionError, match="No tokens found in common with vocabulary"):
        InMemoryTokenizedDataset(
            adata=adata,
            tokenizer=tokenizer
        )


def test_inmemory_tokenized_dataset_sparse_data():
    """Test InMemoryTokenizedDataset with sparse data"""
    # Create AnnData with sparse data
    n_obs, n_vars = 5, 3
    X_sparse = csr_matrix(np.random.rand(n_obs, n_vars))
    var_names = ['gene1', 'gene2', 'gene3']
    
    adata = ad.AnnData(X=X_sparse)
    adata.var_names = var_names
    
    gene_mapping = {
        '<pad>': 0,
        '<cls>': 1,
        'gene1': 2,
        'gene2': 3,
        'gene3': 4,
    }
    tokenizer = GeneIdTokenizer(gene_mapping)
    
    dataset = InMemoryTokenizedDataset(
        adata=adata,
        tokenizer=tokenizer,
        normalization='log1p'
    )
    
    # Test that sparse data is handled correctly
    item = dataset[0]
    assert isinstance(item['values'], np.ndarray)
    assert len(item['values']) == 3  # All genes should be valid