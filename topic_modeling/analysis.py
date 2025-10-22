
import csv
import json
import pandas as pd
import numpy as np
import logging
from typing import Dict, Iterable, List, Optional, Tuple

from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from utils.helpers import maybe_tqdm

def topic_top_words_table(model, feature_names: Iterable[str], n_top_words: int = 12, progress: bool = False) -> pd.DataFrame:
    """Return a table of top terms per topic.

    Args:
        model: Fitted LDA or NMF with .components_.
        feature_names: Vocabulary list/array.
        n_top_words: Number of terms per topic.
        progress: Whether to show a tqdm bar.

    Returns:
        DataFrame with columns [topic, rank, term, weight].

    Input expectations:
        model.components_.shape == (n_topics, n_terms).
    Output expectations:
        Rows == n_topics * n_top_words.
    """
    rows = []
    for k, comp in maybe_tqdm(list(enumerate(model.components_)), enable=progress, desc="Top terms per topic"):
        top_idx = np.argsort(comp)[::-1][:n_top_words]
        for rank, i in enumerate(top_idx, 1):
            rows.append({"topic": k, "rank": rank, "term": feature_names[i], "weight": float(comp[i])})
    return pd.DataFrame(rows)


def top_topics_per_doc(W: np.ndarray, top_n: int = 3, progress: bool = False) -> List[List[Tuple[int, float]]]:
    """Return top-N topics per document from a doc-topic matrix.

    Args:
        W: Document-topic matrix (docs x topics).
        top_n: Number of topics to keep per doc.
        progress: Whether to show a tqdm bar.

    Returns:
        List of lists of (topic_index, weight).

    Input expectations:
        2D array-like with shape (n_docs, n_topics).
    Output expectations:
        Outer list length == n_docs; each inner list length == top_n.
    """
    out: List[List[Tuple[int, float]]] = []
    for row in maybe_tqdm(W, enable=progress, desc="Top topics per doc"):
        top = sorted(enumerate(row), key=lambda x: x[1], reverse=True)[:top_n]
        out.append([(int(i), float(v)) for i, v in top])
    return out


def export_topic_distributions(result: Dict[str, object], out_prefix: str = "topics", logger: Optional[logging.Logger] = None) -> None:
    """Export doc-topic distributions and metrics to CSV/JSON.

    Args:
        result: Output dict from fit_topic_models.
        out_prefix: Filename prefix.
        logger: Optional logger.

    Returns:
        None

    Input expectations:
        Valid result keys present.
    Output expectations:
        Files: <prefix>_lda_doc_topic.csv, <prefix>_nmf_doc_topic.csv, <prefix>_metrics.json.
    """
    names = result.get("doc_names", [f"doc_{i}" for i in range(result["lda_doc_topic"].shape[0])])

    df_lda = pd.DataFrame(result["lda_doc_topic"], columns=[f"lda_topic_{k}" for k in range(result["lda_model"].n_components)])
    df_nmf = pd.DataFrame(result["nmf_doc_topic"], columns=[f"nmf_topic_{k}" for k in range(result["nmf_model"].n_components)])

    df_lda.insert(0, "doc_name", names)
    df_nmf.insert(0, "doc_name", names)

    df_lda.to_csv(f"{out_prefix}_lda_doc_topic.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    df_nmf.to_csv(f"{out_prefix}_nmf_doc_topic.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    with open(f"{out_prefix}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result["metrics"], f, ensure_ascii=False, indent=2)

    if logger:
        logger.info(f"Exports written with prefix '{out_prefix}'")


def export_top_terms(
    lda_model: LatentDirichletAllocation,
    nmf_model: NMF,
    count_vectorizer: CountVectorizer,
    tfidf_vectorizer: TfidfVectorizer,
    out_prefix: str = "topics",
    n_top_words: int = 12,
    progress: bool = False,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Export top terms per topic for LDA and NMF.

    Args:
        lda_model: Fitted LDA.
        nmf_model: Fitted NMF.
        count_vectorizer: Fitted CountVectorizer.
        tfidf_vectorizer: Fitted TfidfVectorizer.
        out_prefix: Filename prefix.
        n_top_words: Terms per topic to export.
        progress: Show tqdm progress bars.
        logger: Optional logger.

    Returns:
        None

    Input expectations:
        Models fitted and vectorizers expose get_feature_names_out.
    Output expectations:
        Files: <prefix>_lda_top_terms.csv, <prefix>_nmf_top_terms.csv.
    """
    if logger:
        logger.info("Computing top terms tables")
    lda_terms = topic_top_words_table(lda_model, count_vectorizer.get_feature_names_out(), n_top_words=n_top_words, progress=progress)
    nmf_terms = topic_top_words_table(nmf_model, tfidf_vectorizer.get_feature_names_out(), n_top_words=n_top_words, progress=progress)

    lda_terms.to_csv(f"{out_prefix}_lda_top_terms.csv", index=False)
    nmf_terms.to_csv(f"{out_prefix}_nmf_top_terms.csv", index=False)
    if logger:
        logger.info(f"Top terms exported with prefix '{out_prefix}'")
