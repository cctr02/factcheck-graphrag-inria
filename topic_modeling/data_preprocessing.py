import re
try:  # minimal dependency handling
    from nltk.corpus import stopwords as nltk_stop
    _ = nltk_stop.words("english")
except Exception:  # lazy download
    import nltk
    nltk.download("stopwords")
    from nltk.corpus import stopwords as nltk_stop


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
    extra.update({"news", "article", "story", "would", "people", "amp", "new"})
    return base.union(extra)
