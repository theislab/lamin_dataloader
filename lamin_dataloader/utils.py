import numpy as np
import logging
logger = logging.getLogger(__name__)
from scipy.sparse import issparse

EPSILON = 1e-3
TOTAL_COUNT = 1e4

def is_raw_count(X):
    if isinstance(X, np.ndarray):
        return np.all(X >= 0) and (X == X.astype(int)).all()
    if issparse(X):
        return (X.data >= 0).all() and (X.data == X.data.astype(int)).all()

def normalize(X, method):
    assert isinstance(X, np.ndarray)
    if method == 'raw':
        X = X
    elif method == 'log1p':
        X = np.log1p(X)
    elif method == 'totalcount':
        X = X / (X.sum(axis=1)[:,None] + EPSILON) * TOTAL_COUNT
    elif method == 'totalcount/log1p' or method == 'totalcount_log1p':
        X = X / (X.sum(axis=1)[:,None] + EPSILON) * TOTAL_COUNT
        X = np.log1p(X)
    elif 'binning' in method: # e.g. binning/100
        n_bins = int(method.split('/')[1])
        X = binning(X, n_bins)
    else:
        raise ValueError(f'Invalid normalization: {method}')
    return X


def binning(row: np.ndarray, n_bins: int) -> np.ndarray:
    """Binning the row into n_bins."""

    if row.max() == 0:
        logger.warning(
            "The input data contains row of zeros. Please make sure this is expected."
        )
        return np.zeros_like(row, dtype=row.dtype)

    non_zero_ids = row.nonzero()
    non_zero_row = row[non_zero_ids]
    non_zero_row = non_zero_row + np.random.random(size=len(non_zero_row)) * 0.01
    bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins+1), method="nearest")
    bins = np.sort(np.unique(bins))
    bins[-1] = bins[-1] + 1.0
    non_zero_digits = np.digitize(non_zero_row, bins)
    assert non_zero_digits.min() >= 1
    assert non_zero_digits.max() <= n_bins
    binned_row = np.zeros_like(row, dtype=row.dtype)
    binned_row[non_zero_ids] = non_zero_digits
    return binned_row
