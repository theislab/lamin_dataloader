"""
Tests for the dataset functionality
"""
import pytest
import numpy as np
from lamin_dataloader.dataset import TokenizedDataset, GeneIdTokenizer
from lamin_dataloader.utils import normalize

class MockCollection:
    """A mock collection class for testing TokenizedDataset functionality."""
    def __init__(self):
        self.var_list = [['gene1', 'gene2', 'gene_not_in_vocab'], ['gene2', 'gene3', 'gene_not_in_vocab']]
        self._path_list = ['path1', 'path2']
        self.n_obs_list = [10, 5]
    
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