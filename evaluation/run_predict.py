import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from .answer_extraction import extract_answer
    from .paths import default_prediction_path
    from .promptor import Promptor
    from .referencer import Referencer
    from .utils import DATA_TYPE_MAPPING
except ImportError:
    from answer_extraction import extract_answer
    from paths import default_prediction_path
    from promptor import Promptor
    from referencer import Referencer
    from utils import DATA_TYPE_MAPPING


DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_BACKEND = "sglang"


def _chunks(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


class BasePredictor:
    backend_name = "base"

    def generate_batch(self, prompts: list[str], max_new_tokens: int) -> list[dict]:
        raise NotImplementedError

    def shutdown(self):
        pass


class SGLangPredictor(BasePredictor):
    backend_name = "sglang"

    def __init__(self, model_path: str, tp_size: int = 1):
        import sglang as sgl
        from transformers import AutoTokenizer

        print(f"Loading SGLang engine from {model_path}")
        self.engine = sgl.Engine(model_path=model_path, tp_size=tp_size)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def _token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def generate_batch(self, prompts: list[str], max_new_tokens: int) -> list[dict]:
        sampling_params = {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
        }
        outputs = self.engine.generate(prompts, sampling_params)
        results = []
        for prompt, output in zip(prompts, outputs):
            if isinstance(output, dict):
                text = output.get("text", "")
            else:
                text = getattr(output, "text", "")
            text = text.strip()
            results.append({
                "text": text,
                "input_tokens": self._token_count(prompt),
                "output_tokens": self._token_count(text),
            })
        return results

    def shutdown(self):
        if hasattr(self.engine, "shutdown"):
            self.engine.shutdown()


class TransformersPredictor(BasePredictor):
    backend_name = "transformers"

    def __init__(self, model_path: str, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading Transformers model from {model_path}")
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {}
        if self.device == "cuda":
            model_kwargs.update({"torch_dtype": torch.float16, "device_map": "cuda"})

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()

    def generate_batch(self, prompts: list[str], max_new_tokens: int) -> list[dict]:
        results = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            input_tokens = int(inputs["input_ids"].shape[-1])
            generated_ids = outputs[0][input_tokens:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            results.append({
                "text": text,
                "input_tokens": input_tokens,
                "output_tokens": int(generated_ids.shape[-1]),
            })
        return results


def create_predictor(model_path: str, backend: str, device: str | None = None, tp_size: int = 1) -> BasePredictor:
    backend = backend.lower()
    if backend == "sglang":
        return SGLangPredictor(model_path, tp_size=tp_size)
    if backend == "transformers":
        return TransformersPredictor(model_path, device=device)
    raise ValueError(f"Unknown backend: {backend}. Use 'sglang' or 'transformers'.")


class TaskProcessor:
    def __init__(self, model: BasePredictor, dataset: str):
        self.model = model
        self.dataset = dataset
        self.promptor = Promptor(dataset)
        self.referencer = Referencer(dataset)
        self.data_type = DATA_TYPE_MAPPING.get(dataset, "MultiChoice")

    def parse_output(self, output: str) -> str:
        return extract_answer(output, self.data_type)

    def _iter_batches(self, prompts: list[str], batch_size: int, progressor):
        batches = list(_chunks(prompts, batch_size))
        if progressor is not None:
            return progressor.tqdm(batches, desc="Evaluating batches")
        if tqdm is not None:
            return tqdm(batches, desc="Evaluating batches")
        return batches

    def predict(self, batch_size: int, max_new_tokens: int, progressor=None):
        predictions = []
        raw_outputs = []
        latencies = []
        input_token_nums = []
        output_token_nums = []

        print("Building prompts...")
        prompts = self.promptor.build_prompt()
        for batch in self._iter_batches(prompts, batch_size, progressor):
            start = time.perf_counter()
            generations = self.model.generate_batch(batch, max_new_tokens=max_new_tokens)
            end = time.perf_counter()
            batch_latency_per_sample = ((end - start) * 1000) / max(1, len(batch))

            for generation in generations:
                output = generation["text"]
                pred = self.parse_output(output)
                print("model output:", pred)

                predictions.append(pred)
                raw_outputs.append(output)
                latencies.append(batch_latency_per_sample)
                input_token_nums.append(generation["input_tokens"])
                output_token_nums.append(generation["output_tokens"])

        print("Building references...")
        references = self.referencer.build_refernce()
        return predictions, references, raw_outputs, latencies, input_token_nums, output_token_nums


def predict(
    dataset: str,
    model_path: str,
    task_type: str | None = None,
    output: str | None = None,
    progressor=None,
    backend: str = DEFAULT_BACKEND,
    batch_size: int = 8,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    device: str | None = None,
    tp_size: int = 1,
):
    data_type = task_type or DATA_TYPE_MAPPING.get(dataset)
    if data_type is None:
        raise ValueError("Unknown dataset type. Pass --data_type explicitly.")

    model = create_predictor(model_path=model_path, backend=backend, device=device, tp_size=tp_size)
    try:
        processor = TaskProcessor(model, dataset)
        predictions, references, raw_outputs, latencies, input_token_nums, output_token_nums = processor.predict(
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            progressor=progressor,
        )
    finally:
        model.shutdown()

    output_data = {
        "predictions": predictions,
        "references": references,
        "raw_outputs": raw_outputs,
        "latencies": latencies,
        "input_lengths": input_token_nums,
        "output_lengths": output_token_nums,
        "lengths": output_token_nums,
        "manifest": {
            "dataset": dataset,
            "data_type": data_type,
            "model_path": model_path,
            "backend": backend,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "num_samples": len(predictions),
            "generation": {
                "do_sample": False,
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "batch_size": batch_size,
                "tp_size": tp_size,
                "latency_note": "Batch latency is divided evenly across samples in each batch.",
            },
        },
    }

    output_path = Path(output) if output else default_prediction_path(dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Prediction results saved to {output_path}")

    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--output", type=str, default=None, help="Prediction output path")
    parser.add_argument("--data_type", type=str, default=None, help="Dataset task type")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="Model path")
    parser.add_argument("--backend", choices=["sglang", "transformers"], default=DEFAULT_BACKEND)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size for the SGLang backend")
    parser.add_argument("--device", type=str, default=None, help="Only used by the transformers backend")

    args = parser.parse_args()
    predict(
        dataset=args.dataset,
        model_path=args.model_path,
        task_type=args.data_type,
        output=args.output,
        backend=args.backend,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        tp_size=args.tp_size,
    )
