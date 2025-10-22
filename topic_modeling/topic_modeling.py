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
    scikit-learn pandas numpy nltk tqdm
"""

from __future__ import annotations
import os
import argparse
import json
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import mean_squared_error

from utils.logger import setup_logger
from utils.helpers import maybe_tqdm
from utils.io import load_corpus_from_csv
from topic_modeling.data_preprocessing import basic_clean, build_stopwords_english
from topic_modeling.analysis import export_topic_distributions, export_top_terms, top_topics_per_doc

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

    out_dir = os.path.join("topic_modeling", args.out)
    os.makedirs(out_dir, exist_ok=True)
    out_prefix = os.path.join(out_dir, args.out)
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
