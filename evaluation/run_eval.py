import json
import argparse
from evaluator import Evaluator
from utils import DATA_TYPE_MAPPING


def load_data(json_file):
    """
    从 JSON 文件加载预测与参考答案
    格式要求：
    ```json
    {
        "predictions": [...],
        "references": [...]
    }
    ```
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['predictions'], data['references'], data['latencies'], data['lengths']


def eval(dataset:str, prednref, metrics:list[str]=None, task_type:str=None, output:str=None):
    data_type = task_type or DATA_TYPE_MAPPING.get(dataset)

    if data_type is None:
        raise ValueError(f"数据集类型未知，请通过 --data_type 指定")

    if isinstance(prednref, str):
        predictions, references, latencies, token_nums = load_data(prednref)
    elif isinstance(prednref, dict):
        predictions = prednref['predictions']
        references = prednref['references']
        latencies = prednref['latencies']
        token_nums = prednref['lengths']
    else:
        raise ValueError('Unknown data type for `prednref`')

    result = Evaluator.evaluate(predictions, references, data_type=data_type, metrics=metrics)
    lpt = [l / t for l, t in zip(latencies, token_nums)]
    result['Latency per token (ms)'] = sum(lpt) / len(lpt)
    print(f"数据集名称: {dataset}")
    print(f"数据集类型: {data_type}")
    print("评估结果: ")
    for metric, score in result.items():
        print(f"  - {metric}: {score}")
    if output:
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        print(f"已将保存评估结果到: {output}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="数据集名称")
    parser.add_argument("--prednref", type=str, required=True, help="输入数据文件路径, JSON 格式, 含 predictions & references")
    parser.add_argument("--output", type=str, default=None, help="（可选）保存评估结果的路径")
    parser.add_argument("--data_type", type=str, default=None)

    args = parser.parse_args()
    eval(args.dataset, args.prednref, task_type=args.data_type, output=args.output)

# CUDA_VISIBLE_DEVICES=0 python run_eval.py --dataset StrategyQA_short --input StrategyQA_short.json