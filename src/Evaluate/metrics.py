"""
Common LLM evaluation metrics: ROUGE, BLEU, BERTScore.
Used to compare model predictions against reference completions.
"""
from __future__ import annotations

import logging
from typing import Sequence

logger = logging.getLogger(__name__)

# ROUGE
try:
    from rouge_score import rouge_scorer
    _ROUGE_AVAILABLE = True
except ImportError:
    _ROUGE_AVAILABLE = False
    rouge_scorer = None

# BLEU
try:
    import sacrebleu
    _SACREBLEU_AVAILABLE = True
except ImportError:
    _SACREBLEU_AVAILABLE = False
    sacrebleu = None

# BERTScore
try:
    import bert_score
    _BERTSCORE_AVAILABLE = True
except ImportError:
    _BERTSCORE_AVAILABLE = False
    bert_score = None


ROUGE_KEYS = ("rouge1", "rouge2", "rougeL")


def compute_rouge(
    references: Sequence[str],
    predictions: Sequence[str],
    use_stemmer: bool = True,
    rouge_keys: tuple[str, ...] = ROUGE_KEYS,
) -> dict[str, float]:
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L (F1) averaged over the corpus.
    """
    if not _ROUGE_AVAILABLE:
        logger.warning("rouge_score not installed; pip install rouge-score")
        return {f"rouge_{k}": 0.0 for k in rouge_keys}

    scorer = rouge_scorer.RougeScorer(list(rouge_keys), use_stemmer=use_stemmer)
    agg: dict[str, list[float]] = {k: [] for k in rouge_keys}
    for ref, pred in zip(references, predictions):
        ref = ref or " "
        pred = pred or " "
        s = scorer.score(ref, pred)
        for k in rouge_keys:
            agg[k].append(s[k].fmeasure)
    return {f"rouge_{k}": sum(agg[k]) / len(agg[k]) if agg[k] else 0.0 for k in rouge_keys}


def compute_bleu(
    references: Sequence[str],
    predictions: Sequence[str],
) -> dict[str, float]:
    """
    Compute corpus BLEU (and chrF2 if available) via sacrebleu.
    references: list of reference strings (one per sample).
    predictions: list of hypothesis strings.
    """
    if not _SACREBLEU_AVAILABLE:
        logger.warning("sacrebleu not installed; pip install sacrebleu")
        return {"bleu": 0.0, "chrf": 0.0}

    # sacrebleu: ref_streams = list of reference streams; one stream = list of ref strings (one per segment)
    ref_streams = [list(references)]
    try:
        bleu = sacrebleu.corpus_bleu(predictions, ref_streams)
        try:
            chrf = sacrebleu.CHRF().corpus_score(predictions, ref_streams)
            return {"bleu": bleu.score / 100.0, "chrf": chrf.score / 100.0}
        except Exception:
            return {"bleu": bleu.score / 100.0, "chrf": 0.0}
    except Exception as e:
        logger.warning("sacrebleu computation failed: %s", e)
        return {"bleu": 0.0, "chrf": 0.0}


def compute_bertscore(
    references: Sequence[str],
    predictions: Sequence[str],
    lang: str = "en",
    model_type: str | None = None,
) -> dict[str, float]:
    """
    Compute BERTScore (P, R, F1) averaged over the corpus.
    """
    if not _BERTSCORE_AVAILABLE:
        logger.warning("bert_score not installed; pip install bert-score")
        return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}

    refs = [r or " " for r in references]
    preds = [p or " " for p in predictions]
    try:
        P, R, F1 = bert_score.score(
            preds, refs, lang=lang, model_type=model_type, verbose=False
        )
        return {
            "bertscore_precision": P.mean().item(),
            "bertscore_recall": R.mean().item(),
            "bertscore_f1": F1.mean().item(),
        }
    except Exception as e:
        logger.warning("bert_score computation failed: %s", e)
        return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}


def compute_all(
    references: Sequence[str],
    predictions: Sequence[str],
    rouge: bool = True,
    bleu: bool = True,
    bertscore: bool = True,
    bertscore_lang: str = "en",
) -> dict[str, float]:
    """Compute all requested metrics and return a single flat dict."""
    out: dict[str, float] = {}
    if rouge:
        out.update(compute_rouge(references, predictions))
    if bleu:
        out.update(compute_bleu(references, predictions))
    if bertscore:
        out.update(compute_bertscore(references, predictions, lang=bertscore_lang))
    return out
