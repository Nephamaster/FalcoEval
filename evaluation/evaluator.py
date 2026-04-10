from typing import List, Union, Set, Dict
from metrics.em_f1 import EM
from metrics.em_f1 import F1
from metrics.accuracy import evaluate as Accuracy
from metrics.accuracy import token_efficiency as TE
from metrics.rouge_bleu import ROUGE, BLEU
from metrics.span_metric import PnRnF


class Evaluator:
    """
    根据任务类型分发至对应指标评估模块
    """

    @staticmethod
    def evaluate(
        predictions: List[Union[str, int, Set[int]]],
        references: List[Union[str, int, List[Union[str, int]], Set[int]]],
        data_type: str,
        metrics:Union[List[str],None],
    ) -> Dict[str, float]:
        """
        通用评估接口

        参数：
        - predictions: 模型预测输出
        - references: 参考答案（可为单个或多个）
        - data_type: 数据集类型标识，如：
            - "Extraction"：抽取式问答(EM/F1)
            - "MultiChoice"：选择题(Accuracy)
            - "Judgement": 判断题(Accuracy)
            - "Math": 数学题(Accuracy)
            - "Generation": 自由文本生成(ROUGE/BLEU)
            - "SPAN": span 索引抽取任务(P/R/F1)
            - "RelExtract": 关系抽取(P/R/F1)

        返回：
        - 指标字典，如 {"F1": 84.2, "EM": 76.4}
        """
        
        if metrics is not None:
            result = {}
            for m in metrics:
                if m == "EM": result[m] = EM(predictions, references)
                elif m == "F1": result[m] = F1(predictions, references)
                elif m == "BLEU": result[m] = BLEU(predictions, references)
                elif m == "ROUGE": result[m] = ROUGE(predictions, references)
                elif m == "Accuracy": result[m] = Accuracy(predictions, references)
                elif m == "TokenEfficiency": result[m] = TE(predictions, references)
                elif m == "Precision": result[m] = PnRnF(predictions, references)
                else: print('Unknown metrics: ', m)
            return result
            
        # if data_type in ["Extraction", "Judgement"]:
        #     return EM_F1(predictions, references)

        # elif data_type in ["MultiChoice", "Math", "RelExtract"]:
        #     return Accuracy(predictions, references)

        # elif data_type == "Generation":
        #     return ROUGE_BLEU(predictions, references)

        # elif data_type in ["SPAN"]:
        #     return SPAN(predictions, references)

        # else:
        #     raise ValueError(f"未知的数据集类型：{data_type}")

