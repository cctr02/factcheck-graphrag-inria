from typing import List, Optional, Tuple
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
