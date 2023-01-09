import os

from ckiptagger import WS
from rouge import Rouge



ws = WS("./data")


def tokenize_and_join(sentences):
    return [" ".join(toks) for toks in ws(sentences)]


rouge = Rouge()


def get_rouge(preds, refs, avg=True, ignore_empty=False):
    """wrapper around: from rouge import Rouge
    Args:
        preds: string or list of strings
        refs: string or list of strings
        avg: bool, return the average metrics if set to True
        ignore_empty: bool, ignore empty pairs if set to True
    """
    if not isinstance(preds, list):
        preds = [preds]
    if not isinstance(refs, list):
        refs = [refs]
    preds, refs = tokenize_and_join(preds), tokenize_and_join(refs)
    return rouge.get_scores(preds, refs, avg=avg, ignore_empty=ignore_empty)
