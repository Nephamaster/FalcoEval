from importlib import import_module
from typing import Dict, List, Set, Union


DEFAULT_METRICS = {
    "Extraction": ["EM", "F1"],
    "Generation": ["ROUGE", "BLEU"],
    "MultiChoice": ["Accuracy"],
    "Judgement": ["Accuracy"],
    "Math": ["Accuracy"],
    "Precision": ["Precision"],
}


def _metric_func(module_name: str, attr: str):
    prefix = f"{__package__}." if __package__ else ""
    module = import_module(f"{prefix}metrics.{module_name}")
    return getattr(module, attr)


class Evaluator:
    @staticmethod
    def evaluate(
        predictions: List[Union[str, int, Set[int]]],
        references: List[Union[str, int, List[Union[str, int]], Set[int]]],
        data_type: str,
        metrics: Union[List[str], None],
    ) -> Dict[str, float]:
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length.")

        metrics = metrics or DEFAULT_METRICS.get(data_type)
        if metrics is None:
            raise ValueError(f"No default metrics configured for data_type={data_type}")

        result = {}
        for metric in metrics:
            if metric == "EM":
                result[metric] = _metric_func("em_f1", "EM")(predictions, references)
            elif metric == "F1":
                result[metric] = _metric_func("em_f1", "F1")(predictions, references)
            elif metric == "BLEU":
                result[metric] = _metric_func("rouge_bleu", "BLEU")(predictions, references)
            elif metric == "ROUGE":
                result[metric] = _metric_func("rouge_bleu", "ROUGE")(predictions, references)
            elif metric == "Accuracy":
                result[metric] = _metric_func("accuracy", "evaluate")(predictions, references)
            elif metric == "TokenEfficiency":
                result[metric] = _metric_func("accuracy", "token_efficiency")(predictions, references)
            elif metric == "Precision":
                result[metric] = _metric_func("span_metric", "PnRnF")(predictions, references)
            else:
                raise ValueError(f"Unknown metric: {metric}")

        return result
