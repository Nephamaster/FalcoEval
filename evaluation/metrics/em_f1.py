import re
import string
from collections import Counter


def normalize_answer(s):
    """对答案进行标准化，去掉标点、大小写、冠词、多余空格等"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact(a_pred, a_gold):
    """判断预测是否与任一参考答案完全匹配"""
    return int(normalize_answer(a_pred) == normalize_answer(a_gold))


def compute_f1(a_pred, a_gold):
    """计算单个预测与参考答案之间的 F1"""
    pred_tokens = normalize_answer(a_pred).split()
    gold_tokens = normalize_answer(a_gold).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        # 空答案处理
        return int(pred_tokens == gold_tokens)
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def EM(predictions, references):
    """
    计算多个预测-参考对的 EM 和 F1
    - predictions: List[str]，预测答案
    - references: List[Union[str, List[str]]]，参考答案（可能是多个）
    """
    total = len(predictions)
    exact_scores = []
    
    for pred, gold in zip(predictions, references):
        if isinstance(gold, list):
            # 多个参考答案，取最大分数
            exact = max(compute_exact(pred, g) for g in gold)
        else:
            exact = compute_exact(pred, gold)
        exact_scores.append(exact)

    em = 100.0 * sum(exact_scores) / total

    return round(em, 2)


def F1(predictions, references):
    """
    计算多个预测-参考对的 EM 和 F1
    - predictions: List[str]，预测答案
    - references: List[Union[str, List[str]]]，参考答案（可能是多个）
    """
    total = len(predictions)
    f1_scores = []

    for pred, gold in zip(predictions, references):
        if isinstance(gold, list):
            # 多个参考答案，取最大分数
            f1 = max(compute_f1(pred, g) for g in gold)
        else:
            f1 = compute_f1(pred, gold)

        f1_scores.append(f1)

    f1 = 100.0 * sum(f1_scores) / total

    return round(f1, 2)