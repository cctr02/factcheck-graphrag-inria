from typing import List, Optional, Tuple, Dict, Any
import logging
import pandas as pd

def load_corpus_from_csv(
    path: str,
    id_col: str = "id",
    title_col: str = "title",
    body_col: str = "body",
    lang_col: str = "language",
    lang_value: str = "english",
    encoding: str = "utf-8",
    n_rows: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    return_metadata: bool = False,
    review_date_col: str = "review_date",
) -> Tuple[List[str], List[str], Optional[List[Dict[str, Any]]]]:
    """Load and combine title + body from a CSV; filter by language.

    Args:
        path: Path to CSV file
        id_col: Column name for document ID
        title_col: Column name for title
        body_col: Column name for body text
        lang_col: Column name for language
        lang_value: Language to filter by
        encoding: File encoding
        n_rows: Maximum number of rows to load
        logger: Optional logger instance
        return_metadata: If True, return document metadata (title, body, review_date, language)
        review_date_col: Column name for review date

    Returns:
        Tuple of (texts, doc_names) or (texts, doc_names, metadata_list) if return_metadata=True
        metadata_list contains dicts with keys: title, body, text, review_date, language
    """

    if logger:
        logger.info(f"Loading CSV: {path}")
    df = pd.read_csv(path, encoding=encoding, nrows=n_rows)

    # Check required columns (body_col is optional)
    required_cols = [id_col, title_col, lang_col]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}'. Found: {list(df.columns)}")

    # Check if body_col exists, if not, we'll use title_col only
    has_body_col = body_col in df.columns

    # compute total row count from the full file (excluding header)
    with open(path, "r", encoding=encoding, errors="ignore") as f:
        before = max(sum(1 for _ in f) - 1, 0)

    df = df[df[lang_col].astype(str).str.lower() == str(lang_value).lower()].copy()
    after = len(df)
    if logger:
        logger.info(f"Filtered language=={lang_value!r}: kept {after}/{before} rows")

    def _combine(row) -> str:
        t = str(row.get(title_col, "") or "").strip()
        if has_body_col:
            b = str(row.get(body_col, "") or "").strip()
            if t and b:
                return f"{t}. {b}"
            return t or b
        return t

    texts = df.apply(_combine, axis=1).astype(str).tolist()
    doc_names = df[id_col].astype(str).tolist()
    if logger:
        logger.info(f"Prepared {len(texts)} documents")

    if return_metadata:
        metadata_list = []
        for _, row in df.iterrows():
            title = str(row.get(title_col, "") or "").strip()
            if has_body_col:
                body = str(row.get(body_col, "") or "").strip()
                # text = title + newline + body
                if title and body:
                    text = f"{title}\n{body}"
                else:
                    text = title or body
            else:
                body = ""
                text = title

            review_date = str(row.get(review_date_col, "") or "").strip() if review_date_col in df.columns else ""
            language = str(row.get(lang_col, "") or "").strip()

            metadata_list.append({
                "title": title,
                "body": body,
                "text": text,
                "review_date": review_date,
                "language": language
            })
        return texts, doc_names, metadata_list

    return texts, doc_names, None


def load_claims_from_csv(
    path: str,
    doc_id_col: str = "id",
    claim_id_col: str = "claim_id",
    claim_col: str = "claim",
    claimant_col: str = "claimant",
    claimant_norm_col: str = "claimant_norm",
    claim_lang_col: str = "claim_lang",
    review_date_col: str = "review_date",
    rating_col: str = "rating",
    fc_lang_col: str = "fc_lang",
    lang_value: str = "english",
    encoding: str = "utf-8",
    n_rows: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """Load claims from CSV and create mapping to documents.

    Args:
        path: Path to claim_reviews CSV file
        doc_id_col: Column name for document ID
        claim_id_col: Column name for claim ID
        claim_col: Column name for claim text
        claimant_col: Column name for claimant
        claimant_norm_col: Column name for normalized claimant
        claim_lang_col: Column name for claim language
        review_date_col: Column name for review date
        rating_col: Column name for rating
        fc_lang_col: Column name for fact-check language (used for filtering)
        lang_value: Language to filter by (on fc_lang_col)
        encoding: File encoding
        n_rows: Maximum number of rows to load
        logger: Optional logger instance

    Returns:
        Tuple of (claims_list, claim_to_docs_mapping)
        - claims_list: List of claim dicts with keys: claim_id, text, rating, claimant,
          claimant_norm, review_date, language
        - claim_to_docs_mapping: Dict mapping claim_id to list of doc_ids
    """
    if logger:
        logger.info(f"Loading claims from CSV: {path}")

    df = pd.read_csv(path, encoding=encoding, nrows=n_rows)

    # Check required columns exist
    required_cols = [doc_id_col, claim_id_col, claim_col, fc_lang_col]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}'. Found: {list(df.columns)}")

    # Filter by fact-check language
    before = len(df)
    df = df[df[fc_lang_col].astype(str).str.lower() == str(lang_value).lower()].copy()
    after = len(df)
    if logger:
        logger.info(f"Filtered fc_lang=={lang_value!r}: kept {after}/{before} rows")

    # Group by claim_id to deduplicate claims and gather all doc_ids per claim
    claim_to_docs: Dict[str, List[str]] = {}
    claims_data: Dict[str, Dict[str, Any]] = {}

    for _, row in df.iterrows():
        claim_id = str(row.get(claim_id_col, "")).strip()
        doc_id = str(row.get(doc_id_col, "")).strip()

        if not claim_id or not doc_id:
            continue

        # Add doc_id to mapping
        if claim_id not in claim_to_docs:
            claim_to_docs[claim_id] = []
        if doc_id not in claim_to_docs[claim_id]:
            claim_to_docs[claim_id].append(doc_id)

        # Store claim data (only once per claim)
        if claim_id not in claims_data:
            claim_text = str(row.get(claim_col, "") or "").strip()
            rating = str(row.get(rating_col, "") or "").strip() if rating_col in df.columns else ""
            claimant = str(row.get(claimant_col, "") or "").strip() if claimant_col in df.columns else ""
            claimant_norm = str(row.get(claimant_norm_col, "") or "").strip() if claimant_norm_col in df.columns else ""
            review_date = str(row.get(review_date_col, "") or "").strip() if review_date_col in df.columns else ""
            language = str(row.get(claim_lang_col, "") or "").strip() if claim_lang_col in df.columns else ""

            claims_data[claim_id] = {
                "claim_id": claim_id,
                "text": claim_text,
                "rating": rating,
                "claimant": claimant,
                "claimant_norm": claimant_norm,
                "review_date": review_date,
                "language": language
            }

    claims_list = list(claims_data.values())

    if logger:
        logger.info(f"Loaded {len(claims_list)} unique claims linked to {len(df)} documents")

    return claims_list, claim_to_docs

