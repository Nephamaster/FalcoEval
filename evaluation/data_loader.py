import json
import warnings
from pathlib import Path
from typing import List


def _has_answer(sample: dict) -> bool:
    if "answers" in sample:
        answer = sample["answers"]
    else:
        answer = sample.get("answer")

    if answer is None:
        return False
    if isinstance(answer, str):
        return bool(answer.strip())
    if isinstance(answer, list):
        return any(str(item).strip() for item in answer)
    return True


def _required_fields(data_type: str) -> set[str]:
    if data_type == "MultiChoice":
        return {"question", "choices"}
    if data_type == "WikiEvent":
        return {"target", "entities", "retrieved_sentences"}
    return {"question"}


def load_jsonl_dataset(path: Path, data_type: str, skip_invalid: bool = True) -> List[dict]:
    samples: List[dict] = []
    invalid = 0
    examples = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid += 1
                if len(examples) < 5:
                    examples.append(f"line {line_no}: invalid JSON: {exc}")
                if skip_invalid:
                    continue
                raise

            missing = _required_fields(data_type) - set(sample)
            if missing or not _has_answer(sample):
                invalid += 1
                if len(examples) < 5:
                    examples.append(
                        f"line {line_no}: missing={sorted(missing)}, has_answer={_has_answer(sample)}"
                    )
                if skip_invalid:
                    continue
                raise ValueError(
                    f"Invalid sample at {path}:{line_no}: "
                    f"missing={sorted(missing)}, has_answer={_has_answer(sample)}"
                )

            samples.append(sample)

    if not samples:
        detail = "; ".join(examples)
        raise ValueError(f"No valid samples found in {path}; skipped {invalid} invalid rows. Examples: {detail}")

    if invalid:
        detail = "; ".join(examples)
        warnings.warn(
            f"Loaded {len(samples)} valid samples from {path}; skipped {invalid} invalid rows. Examples: {detail}"
        )

    return samples
