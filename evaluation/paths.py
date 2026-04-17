from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "datasets"
OUTPUTS_DIR = PROJECT_ROOT / "output"


def dataset_path(dataset: str) -> Path:
    path = DATASETS_DIR / f"{dataset}.jsonl"
    if path.exists():
        return path
    if dataset.endswith("_demo"):
        fallback = DATASETS_DIR / f"{dataset[:-5]}_short.jsonl"
        if fallback.exists():
            return fallback
    return path


def default_prediction_path(dataset: str) -> Path:
    return OUTPUTS_DIR / f"{dataset}_pred_ref.json"
