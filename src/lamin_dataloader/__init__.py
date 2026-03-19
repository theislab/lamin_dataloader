"""Lamin DataLoader - A data loading library for AnnData collections."""

import logging
import os
from importlib.metadata import version

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(levelname)s: %(message)s",
)

from lamin_dataloader.collections import Collection, InMemoryCollection
from lamin_dataloader.dataset import (
    Tokenizer,
    GeneIdTokenizer,
    TokenizedDataset,
    BaseCollate,
)
from lamin_dataloader.samplers import SubsetSampler

__all__ = [
    "Collection",
    "InMemoryCollection",
    "Tokenizer",
    "GeneIdTokenizer",
    "TokenizedDataset",
    "BaseCollate",
    "SubsetSampler",
]

__version__ = version("lamin_dataloader")
