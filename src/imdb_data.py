from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split


@dataclass
class ImdbData:
    train_texts: list[str]
    train_labels: np.ndarray
    val_texts: list[str]
    val_labels: np.ndarray
    test_texts: list[str]
    test_labels: np.ndarray


def load_imdb(
    *,
    seed: int = 42,
    val_fraction: float = 0.1,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
) -> ImdbData:
    """
    Loads IMDb and returns train/val/test splits.

    IMDb comes as:
      - train: 25k
      - test: 25k
    We carve out a validation split from the original train split.
    """
    ds = load_dataset("imdb")

    # IMDb labels are 0 (neg) and 1 (pos) already.
    train_split = ds["train"]
    test_split = ds["test"]

    train_texts = np.array(train_split["text"], dtype=object)
    train_labels = np.array(train_split["label"], dtype=np.int64)

    test_texts = np.array(test_split["text"], dtype=object)
    test_labels = np.array(test_split["label"], dtype=np.int64)

    # Optional subsampling for faster iterations.
    if max_test_samples is not None and max_test_samples < len(test_texts):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(test_texts), size=max_test_samples, replace=False)
        test_texts = test_texts[idx]
        test_labels = test_labels[idx]

    # Validation split from the original training split.
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts,
        train_labels,
        test_size=val_fraction,
        random_state=seed,
        stratify=train_labels,
    )

    if max_train_samples is not None and max_train_samples < len(train_texts):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(train_texts), size=max_train_samples, replace=False)
        train_texts = train_texts[idx]
        train_labels = train_labels[idx]

    if max_val_samples is not None and max_val_samples < len(val_texts):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(val_texts), size=max_val_samples, replace=False)
        val_texts = val_texts[idx]
        val_labels = val_labels[idx]

    # Convert back to list for sklearn compatibility.
    return ImdbData(
        train_texts=train_texts.tolist(),
        train_labels=train_labels,
        val_texts=val_texts.tolist(),
        val_labels=val_labels,
        test_texts=test_texts.tolist(),
        test_labels=test_labels,
    )

