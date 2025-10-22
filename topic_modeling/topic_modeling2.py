"""
Topic Modeling Pipeline (CSV with headers: id, url, title, body, language, publisher_name, publisher_site, review_date)

- Data loading from CSV (auto-uses title + body, filters language=='english')
- Minimal cleaning & tokenization
- Vectorization (Count for LDA, TF-IDF for NMF)
- Models: LDA (scikit-learn) & NMF
- Quick inspection: top words per topic, top topics per document
- Quick evaluation: perplexity (LDA) & reconstruction MSE (NMF)
- CSV export of topic distributions & top terms
- Simple CLI with logging and progress bars (tqdm)

Usage examples:
    python topic_modeling/topic_modeling2.py --csv data/english_fact_checks.csv --topics 12 --out topics_run --log-level INFO --progress

Requirements:
    pip install scikit-learn pandas numpy nltk tqdm
"""

from __future__ import annotations

import os
import argparse
import csv
import json
import logging
import re
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import mean_squared_error

# ---- Optional NLTK stopwords (download on first run if missing) ----
try:  # minimal dependency handling
    from nltk.corpus import stopwords as nltk_stop
    _ = nltk_stop.words("english")
except Exception:  # lazy download
    import nltk
    nltk.download("stopwords")
    from nltk.corpus import stopwords as nltk_stop


# ====================================================================
# Logging utilities
# ====================================================================

def setup_logger(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Configure and return a module-level logger.

    Args:
        level: Log level name (e.g., 'DEBUG', 'INFO', 'WARNING').
        log_file: Optional path to a file to also write logs.

    Returns:
        Configured logger instance.

    Input expectations:
        Valid logging level string.
    Output expectations:
        Logger emitting to stdout with time, level, message.
    """
    logger = logging.getLogger("topic_modeling")
    # remove existing handlers to avoid duplicate logs when reconfiguring
    if logger.handlers:
        for h in logger.handlers[:]:
            logger.removeHandler(h)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler()
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt))
        logger.addHandler(fh)

    return logger


def maybe_tqdm(iterable, enable: bool, **kwargs):
    """Wrap iterable with tqdm when enabled.

    Args:
        iterable: Any iterable.
        enable: Whether to show progress bar.

    Returns:
        tqdm-wrapped iterable if enabled, else original iterable.

    Input expectations:
        Iterable object.
    Output expectations:
        Iterable preserving items order; visual progress if enabled.
    """
    return tqdm(iterable, **kwargs) if enable else iterable


# ====================================================================
# I/O
# ====================================================================

def load_corpus_from_csv(
    path: str,
    id_col: str = "id",
    title_col: str = "title",
    body_col: str = "body",
    lang_col: str = "language",
    lang_value: str = "english",
    encoding: str = "utf-8",
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[str], List[str]]:
    """Load and combine title + body from a CSV; filter by language.

    Args:
        path: Path to CSV.
        id_col: Column name for unique ids.
        title_col: Column name for titles.
        body_col: Column name for bodies.
        lang_col: Column name for language.
        lang_value: Language to keep (case-insensitive).
        encoding: File encoding.
        logger: Optional logger.

    Returns:
        (texts, doc_names) where texts are concatenated title + body strings,
        and doc_names are the ids as strings.

    Input expectations:
        CSV must contain the headers listed in the arguments; language rows are filtered to lang_value.
    Output expectations:
        Lists of equal length; len(texts)==len(doc_names)>=0.
    """
    if logger:
        logger.info(f"Loading CSV: {path}")
    df = pd.read_csv(path, encoding=encoding)
    for c in [id_col, title_col, body_col, lang_col]:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}'. Found: {list(df.columns)}")

    before = len(df)
    df = df[df[lang_col].astype(str).str.lower() == str(lang_value).lower()].copy()
    after = len(df)
    if logger:
        logger.info(f"Filtered language=={lang_value!r}: kept {after}/{before} rows")

    def _combine(row) -> str:
        t = str(row.get(title_col, "") or "").strip()
        b = str(row.get(body_col, "") or "").strip()
        if t and b:
            return f"{t}. {b}"
        return t or b

    texts = df.apply(_combine, axis=1).astype(str).tolist()
    doc_names = df[id_col].astype(str).tolist()
    if logger:
        logger.info(f"Prepared {len(texts)} documents")
    return texts, doc_names


# ====================================================================
# Preprocessing
# ====================================================================

def basic_clean(text: str) -> str:
    """Minimal text cleaning.

    Args:
        text: Raw input text.

    Returns:
        Cleaned lowercase text.

    Input expectations:
        Any string.
    Output expectations:
        String with urls/emails/digits removed, trimmed and lowercased.
    """
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[\d]+", " ", text)
    text = re.sub(r"[^\w\s'-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def build_stopwords_english() -> set:
    """Return an English stopword set.

    Args:
        None

    Returns:
        Set of stopwords.

    Input expectations:
        NLTK stopwords available (downloaded if missing).
    Output expectations:
        Non-empty set of strings.
    """
    base = set(nltk_stop.words("english"))
    extra = {"say","says","said","also","one","two","us","u","like","just","get","got","make","made","via"}
    return base.union(extra)


# ====================================================================
# Vectorization
# ====================================================================

def make_vectorizers(
    stopset: set,
    max_df: float = 0.95,
    min_df: int = 2,
    max_features: int = 30000,
    ngram_range: Tuple[int, int] = (1,1)
) -> Tuple[CountVectorizer, TfidfVectorizer]:
    """Create Count and TF-IDF vectorizers.

    Args:
        stopset: Stopword set.
        max_df: Ignore terms in >max_df of docs.
        min_df: Ignore terms in <min_df docs.
        max_features: Vocabulary cap.
        ngram_range: e.g., (1,1) or (1,2).

    Returns:
        (count_vectorizer, tfidf_vectorizer).

    Input expectations:
        stopset convertible to list.
    Output expectations:
        Fitted later on text; not yet fitted here.
    """
    stops = list(stopset)
    count_vec = CountVectorizer(stop_words=stops, max_df=max_df, min_df=min_df, max_features=max_features, ngram_range=ngram_range)
    tfidf_vec = TfidfVectorizer(stop_words=stops, max_df=max_df, min_df=min_df, max_features=max_features, ngram_range=ngram_range, norm="l2", sublinear_tf=True)
    return count_vec, tfidf_vec


# ====================================================================
# Topic printing / inspection
# ====================================================================

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


# ====================================================================
# Modeling
# ====================================================================

def fit_topic_models(
    corpus: List[str],
    n_topics: int = 10,
    max_df: float = 0.95,
    min_df: int = 2,
    max_features: int = 30000,
    ngram_range: Tuple[int, int] = (1,1),
    lda_max_iter: int = 20,
    nmf_max_iter: int = 400,
    random_state: int = 42,
    doc_names: Optional[List[str]] = None,
    progress: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, object]:
    """Fit LDA (on Count) and NMF (on TF-IDF) over a corpus.

    Args:
        corpus: List of raw documents.
        n_topics: Desired number of topics.
        max_df: Vectorizer max_df.
        min_df: Vectorizer min_df.
        max_features: Vocabulary cap.
        ngram_range: e.g., (1,2) to include bigrams.
        lda_max_iter: LDA iterations.
        nmf_max_iter: NMF iterations.
        random_state: RNG seed.
        doc_names: Optional names for documents.
        progress: Show tqdm progress bars.
        logger: Optional logger.

    Returns:
        Dict with cleaned corpus, fitted vectorizers/models, matrices, and metrics.

    Input expectations:
        Non-empty corpus.
    Output expectations:
        Keys: [clean_corpus, count_vectorizer, tfidf_vectorizer, X_count, X_tfidf,
               lda_model, nmf_model, lda_doc_topic, nmf_doc_topic, lda_top_per_doc,
               nmf_top_per_doc, metrics, doc_names].
    """
    if not corpus:
        raise ValueError("Empty corpus.")

    if logger:
        logger.info("Cleaning corpus")
    clean = []
    for t in maybe_tqdm(corpus, enable=progress, desc="Cleaning"):
        clean.append(basic_clean(t))

    stopset = build_stopwords_english()
    if logger:
        logger.info("Building vectorizers")
    count_vec, tfidf_vec = make_vectorizers(stopset, max_df=max_df, min_df=min_df, max_features=max_features, ngram_range=ngram_range)

    if logger:
        logger.info("Fitting CountVectorizer (LDA input)")
    X_count = count_vec.fit_transform(clean)

    if logger:
        logger.info("Fitting TfidfVectorizer (NMF input)")
    X_tfidf = tfidf_vec.fit_transform(clean)

    if not doc_names or len(doc_names) != len(corpus):
        doc_names = [f"doc_{i}" for i in range(len(corpus))]

    if logger:
        logger.info("Training LDA")
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method="batch",
        max_iter=lda_max_iter,
        random_state=random_state,
        n_jobs=-1,
        evaluate_every=0,
    ).fit(X_count)

    if logger:
        logger.info("Transforming documents with LDA")
    lda_doc_topic = lda.transform(X_count)
    lda_perplexity = float(lda.perplexity(X_count))
    if logger:
        logger.info(f"LDA perplexity: {lda_perplexity:.4f}")

    if logger:
        logger.info("Training NMF")
    n_docs, n_terms = X_tfidf.shape
    n_comp_cap = int(min(n_topics, n_docs, n_terms))
    nmf_init = "nndsvda" if n_comp_cap == n_topics else "random"

    nmf = NMF(
        n_components=n_comp_cap,
        init=nmf_init,
        max_iter=nmf_max_iter,
        random_state=random_state,
        alpha_W=0.0,
        alpha_H=0.0,
        l1_ratio=0.0,
        verbose=0,
    ).fit(X_tfidf)

    if logger:
        logger.info("Transforming documents with NMF and computing reconstruction error")
    W = nmf.transform(X_tfidf)
    H = nmf.components_
    recon_err = float(mean_squared_error(X_tfidf.toarray(), np.dot(W, H)))
    if logger:
        logger.info(f"NMF reconstruction MSE: {recon_err:.6f}")

    lda_top = top_topics_per_doc(lda_doc_topic, top_n=3, progress=progress)
    nmf_top = top_topics_per_doc(W, top_n=3, progress=progress)

    metrics = {
        "lda_perplexity": lda_perplexity,
        "nmf_mse_reconstruction": recon_err,
        "n_docs": int(len(corpus)),
        "n_terms_count": int(X_count.shape[1]),
        "n_terms_tfidf": int(X_tfidf.shape[1]),
        "n_topics_requested": int(n_topics),
        "n_topics_nmf": int(nmf.n_components),
    }

    return {
        "clean_corpus": clean,
        "count_vectorizer": count_vec,
        "tfidf_vectorizer": tfidf_vec,
        "X_count": X_count,
        "X_tfidf": X_tfidf,
        "lda_model": lda,
        "nmf_model": nmf,
        "lda_doc_topic": lda_doc_topic,
        "nmf_doc_topic": W,
        "lda_top_per_doc": lda_top,
        "nmf_top_per_doc": nmf_top,
        "metrics": metrics,
        "doc_names": doc_names,
    }


# ====================================================================
# Export utilities
# ====================================================================

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


# ====================================================================
# CLI
# ====================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        None

    Returns:
        argparse.Namespace with parsed args.

    Input expectations:
        Standard CLI usage.
    Output expectations:
        Namespace with fields used in main().
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to CSV with headers including title, body, language, id")
    ap.add_argument("--topics", type=int, default=10, help="Number of topics to learn")
    ap.add_argument("--max-df", type=float, default=0.95)
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("--max-features", type=int, default=30000)
    ap.add_argument("--ngrams", type=str, default="1,1", help="e.g., '1,2' for unigrams+bigrams")
    ap.add_argument("--out", type=str, default="topics", help="Export filename prefix (directory will be topic_modeling/<out>)")
    ap.add_argument("--log-level", type=str, default="INFO", help="Logging level: DEBUG/INFO/WARNING/ERROR")
    # tqdm/progressbars enabled by default
    ap.add_argument("--progress", action="store_true", default=True, help="Show tqdm progress bars (enabled by default)")
    return ap.parse_args()


def main() -> None:
    """Entry point for CLI runs.

    Args:
        None

    Returns:
        None

    Input expectations:
        CSV file exists; language column equals 'english'.
    Output expectations:
        Writes CSV/JSON exports in current directory.
    """
    args = parse_args()

    # prepare output directory under topic_modeling/<out>
    out_dir = os.path.join("topic_modeling", args.out)
    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, f"{args.out}.log")

    logger = setup_logger(args.log_level, log_file=log_file)
    logger.info("Starting topic modeling pipeline")

    texts, names = load_corpus_from_csv(
        path=args.csv,
        id_col="id",
        title_col="title",
        body_col="body",
        lang_col="language",
        lang_value="english",
        logger=logger,
    )

    ngram_range = tuple(int(x) for x in args.ngrams.split(","))

    result = fit_topic_models(
        corpus=texts,
        n_topics=args.topics,
        max_df=args.max_df,
        min_df=args.min_df,
        max_features=args.max_features,
        ngram_range=ngram_range,
        doc_names=names,
        progress=args.progress,
        logger=logger,
    )
    
    # use out_dir for exports
    out_prefix = os.path.join(out_dir, args.out)
    export_topic_distributions(result, out_prefix=out_prefix, logger=logger)
    export_top_terms(
        lda_model=result["lda_model"],
        nmf_model=result["nmf_model"],
        count_vectorizer=result["count_vectorizer"],
        tfidf_vectorizer=result["tfidf_vectorizer"],
        out_prefix=out_prefix,
        progress=args.progress,
        logger=logger,
    )

    logger.info("Pipeline finished. Metrics below:")
    logger.info("\n" + json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
