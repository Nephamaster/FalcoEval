def _normalize(value):
    return str(value).strip().lower()


def _matches(prediction, reference) -> bool:
    prediction = _normalize(prediction)
    if isinstance(reference, list):
        return any(prediction == _normalize(item) for item in reference)
    return prediction == _normalize(reference)


def evaluate(predictions, references):
    """
    Compute exact-match accuracy after lightweight normalization.
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length.")

    correct = sum(_matches(p, r) for p, r in zip(predictions, references))
    total = len(predictions)
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return round(accuracy, 2)


def token_efficiency(predictions, references):
    """
    Accuracy normalized by average output length.

    Runtime token throughput is computed in run_eval.py from model-tokenizer output
    lengths. This metric is kept as a quality/verbosity proxy and avoids loading a
    separate tokenizer with a hard-coded local path.
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length.")

    total = len(predictions)
    if total == 0:
        return {"Generation Length (tokens)": 0.0, "Token Efficiency": 0.0}

    accuracy = evaluate(predictions, references)
    lengths = [max(1, len(str(prediction).split())) for prediction in predictions]
    avg_token_num = sum(lengths) / total
    return {
        "Generation Length (tokens)": round(avg_token_num, 2),
        "Token Efficiency": round(accuracy / avg_token_num, 4),
    }
