from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


def ROUGE(predictions, references):
    """
    计算 ROUGE-L 分数（平均）
    - predictions: List[str]
    - references: List[str] 或 List[List[str]]
    """
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = []

    for pred, ref in zip(predictions, references):
        if isinstance(ref, list):
            scores = [scorer.score(pred, r)['rougeL'].fmeasure for r in ref]
            rouge_scores.append(max(scores))
        else:
            score = scorer.score(pred, ref)['rougeL'].fmeasure
            rouge_scores.append(score)

    avg_rouge = 100.0 * sum(rouge_scores) / len(rouge_scores)
    return round(avg_rouge, 2)


def BLEU(predictions, references):
    """
    计算 BLEU-4 分数（平均）
    - predictions: List[str]
    - references: List[str] 或 List[List[str]]
    """
    smoothie = SmoothingFunction().method4
    bleu_scores = []

    for pred, ref in zip(predictions, references):
        pred_tokens = pred.strip().split()
        if isinstance(ref, list):
            ref_tokens = [r.strip().split() for r in ref]
        else:
            ref_tokens = [ref.strip().split()]
        
        bleu = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothie)
        bleu_scores.append(bleu)

    avg_bleu = 100.0 * sum(bleu_scores) / len(bleu_scores)
    return round(avg_bleu, 2)


def evaluate(predictions, references):
    """
    综合计算 ROUGE-L 和 BLEU 分数
    """
    return {
        'ROUGE-L': ROUGE(predictions, references),
        'BLEU': BLEU(predictions, references)
    }
