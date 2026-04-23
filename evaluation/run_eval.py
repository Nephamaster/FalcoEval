import argparse
import json
from pathlib import Path

try:
    from .evaluator import Evaluator
    from .utils import DATA_TYPE_MAPPING
except ImportError:
    from evaluator import Evaluator
    from utils import DATA_TYPE_MAPPING


def load_data(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def _performance_metrics(latencies, output_lengths):
    latencies = [float(item) for item in latencies]
    output_lengths = [int(item) for item in output_lengths]
    non_zero = [(latency, tokens) for latency, tokens in zip(latencies, output_lengths) if tokens > 0]

    def mean(values):
        return sum(values) / len(values) if values else 0.0

    def percentile(values, percent):
        if not values:
            return 0.0
        values = sorted(values)
        index = (len(values) - 1) * (percent / 100)
        lower = int(index)
        upper = min(lower + 1, len(values) - 1)
        weight = index - lower
        return values[lower] * (1 - weight) + values[upper] * weight

    result = {}
    if latencies:
        result["Latency mean (ms)"] = round(float(mean(latencies)), 2)
        result["Latency p50 (ms)"] = round(float(percentile(latencies, 50)), 2)
        result["Latency p95 (ms)"] = round(float(percentile(latencies, 95)), 2)
    if non_zero:
        per_token = [latency / tokens for latency, tokens in non_zero]
        total_tokens = sum(tokens for _, tokens in non_zero)
        total_seconds = sum(latency for latency, _ in non_zero) / 1000
        result["Latency per output token (ms)"] = round(float(mean(per_token)), 2)
        result["Output tokens/s"] = round(float(total_tokens / total_seconds), 2) if total_seconds else 0.0
        result["Output length mean (tokens)"] = round(float(mean(output_lengths)), 2)
    return result


def _runtime_metrics(runtime):
    if not runtime:
        return {}

    result = {
        "Model reused": bool(runtime.get("reused_model", False)),
        "Cold start load (ms)": round(float(runtime.get("cold_start_load_ms", 0.0)), 2),
        "Warmup batch (ms)": round(float(runtime.get("warmup_batch_ms", 0.0)), 2),
        "Warmup batches": int(runtime.get("warmup_batches", 0)),
        "Warmup samples": int(runtime.get("warmup_samples", 0)),
        "Steady-state generation (ms)": round(float(runtime.get("steady_state_generation_ms", 0.0)), 2),
        "Steady-state samples": int(runtime.get("steady_state_samples", 0)),
        "Total generation (ms)": round(float(runtime.get("total_generation_ms", 0.0)), 2),
    }

    note = runtime.get("latency_scope")
    if note:
        result["Latency scope"] = note
    return result


def eval(dataset: str, prednref, metrics: list[str] | None = None, task_type: str | None = None, output: str | None = None):
    data_type = task_type or DATA_TYPE_MAPPING.get(dataset)
    if data_type is None:
        raise ValueError("Unknown dataset type. Pass --data_type explicitly.")

    if isinstance(prednref, str):
        data = load_data(prednref)
    elif isinstance(prednref, dict):
        data = prednref
    else:
        raise ValueError("Unknown data type for `prednref`")

    predictions = data["predictions"]
    references = data["references"]
    latencies = data.get("latencies", [])
    output_lengths = data.get("output_lengths", data.get("lengths", []))
    runtime = data.get("runtime", {})

    result = Evaluator.evaluate(predictions, references, data_type=data_type, metrics=metrics)
    result.update(_runtime_metrics(runtime))
    result.update(_performance_metrics(latencies, output_lengths))

    print(f"Dataset: {dataset}")
    print(f"Data type: {data_type}")
    print("Evaluation results:")
    for metric, score in result.items():
        print(f"  - {metric}: {score}")

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Evaluation results saved to {output_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--prednref", type=str, required=True, help="Prediction/reference JSON path")
    parser.add_argument("--output", type=str, default=None, help="Optional evaluation output path")
    parser.add_argument("--data_type", type=str, default=None)
    parser.add_argument("--metrics", nargs="*", default=None)

    args = parser.parse_args()
    eval(args.dataset, args.prednref, metrics=args.metrics, task_type=args.data_type, output=args.output)
