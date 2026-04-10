def compute_f1(pred_span, gold_span):
    """
    计算单个样本的 precision, recall, f1
    - pred_span: Set[int]，预测的 token 索引集合
    - gold_span: Set[int]，参考答案的 token 索引集合
    """
    if not pred_span and not gold_span:
        return 1.0, 1.0, 1.0
    if not pred_span or not gold_span:
        return 0.0, 0.0, 0.0

    overlap = pred_span & gold_span
    tp = len(overlap)
    precision = tp / len(pred_span) if pred_span else 0
    recall = tp / len(gold_span) if gold_span else 0
    f1 = 2 * precision * recall / (precision + recall) if tp > 0 else 0.0
    return precision, recall, f1


def PnRnF(predictions, references):
    """
    计算整组样本的平均 Precision, Recall, F1
    - predictions: List[Set[int]]，每条预测的 span 索引集合
    - references: List[Set[int]]，每条参考的 span 索引集合
    """
    assert len(predictions) == len(references), "预测与参考数量不一致"

    total_p, total_r, total_f1 = 0.0, 0.0, 0.0
    n = len(predictions)

    for pred, gold in zip(predictions, references):
        p, r, f1 = compute_f1(set(pred), set(gold))
        total_p += p
        total_r += r
        total_f1 += f1

    avg_p = 100.0 * total_p / n
    avg_r = 100.0 * total_r / n
    avg_f1 = 100.0 * total_f1 / n

    return {
        'Precision': round(avg_p, 2),
        'Recall': round(avg_r, 2),
        'F1': round(avg_f1, 2)
    }
