"""Tests for samplers."""

import numpy as np

from lamin_dataloader.samplers import SubsetSampler


def test_subset_sampler_no_filter():
    """SubsetSampler with no filter uses all indices in range(dataset_size)."""
    dataset_size = 10
    batch_size = 2
    sampler = SubsetSampler(
        dataset_size=dataset_size,
        batch_size=batch_size,
        obs_list_dict={},
        obs_filter_dict=None,
        shuffle=False,
        drop_last=True,
    )
    indices = list(sampler)
    # 5 full batches of 2 (no partial batch to drop)
    assert len(indices) == 10
    assert set(indices) == set(range(10))
    assert len(sampler) == 10


def test_subset_sampler_celltype_filter():
    """SubsetSampler with one filter (celltype) returns only matching indices."""
    dataset_size = 10
    batch_size = 2
    # 4 T_cell at indices 0-3, 6 B_cell at indices 4-9
    celltype = np.array(["T_cell"] * 4 + ["B_cell"] * 6)
    obs_list_dict = {"celltype": celltype}
    obs_filter_dict = {"celltype": np.array(["T_cell"])}

    sampler = SubsetSampler(
        dataset_size=dataset_size,
        batch_size=batch_size,
        obs_list_dict=obs_list_dict,
        obs_filter_dict=obs_filter_dict,
        shuffle=False,
        drop_last=True,
    )
    indices = list(sampler)
    # Only T_cell indices 0,1,2,3 -> 2 full batches of 2
    assert len(indices) == 4
    assert set(indices) == {0, 1, 2, 3}
    assert len(sampler) == 4
