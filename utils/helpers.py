import tqdm

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