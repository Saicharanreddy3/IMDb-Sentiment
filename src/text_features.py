from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def fit_tfidf_svd(
    train_texts: list[str],
    *,
    max_features: int = 40000,
    ngram_range: tuple[int, int] = (1, 2),
    svd_components: int = 300,
    seed: int = 42,
) -> tuple[TfidfVectorizer, TruncatedSVD]:
    """
    Fits a TF-IDF vectorizer and TruncatedSVD for dimensionality reduction.

    We intentionally convert high-dimensional sparse TF-IDF into a dense
    representation (SVD) so we can train models like XGBoost efficiently.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
        dtype=np.float32,
    )

    tfidf_train = vectorizer.fit_transform(train_texts)

    svd = TruncatedSVD(n_components=svd_components, random_state=seed)
    svd.fit(tfidf_train)

    return vectorizer, svd


def transform_to_svd(
    vectorizer: TfidfVectorizer,
    svd: TruncatedSVD,
    texts: list[str],
) -> np.ndarray:
    tfidf = vectorizer.transform(texts)
    X = svd.transform(tfidf)
    # Keep a small memory footprint for downstream training.
    return X.astype(np.float32, copy=False)


def extract_top_terms_for_svd_components(
    vectorizer: TfidfVectorizer,
    svd: TruncatedSVD,
    *,
    top_k: int = 8,
) -> list[list[str]]:
    """
    Produces human-interpretable "top terms per latent SVD component".

    Note: components are latent; this is an approximate explanation by mapping
    component weights back to TF-IDF terms.
    """
    terms = vectorizer.get_feature_names_out()
    comp = svd.components_  # shape: [n_components, n_features]

    top_terms_per_component: list[list[str]] = []
    for i in range(comp.shape[0]):
        weights = comp[i]
        top_idx = np.argsort(np.abs(weights))[::-1][:top_k]
        top_terms_per_component.append([str(terms[j]) for j in top_idx])
    return top_terms_per_component

