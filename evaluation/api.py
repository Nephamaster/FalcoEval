import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    import gradio as gr
except ImportError:  # pragma: no cover - only used when legacy Gradio is not installed.
    gr = None

try:
    from . import main as legacy_ui
    from .utils import DATA_TYPE_MAPPING, dataset_choices, model_paths
except ImportError:
    import main as legacy_ui
    from utils import DATA_TYPE_MAPPING, dataset_choices, model_paths


JobStatus = Literal["queued", "running", "succeeded", "failed"]


class RegisterModelRequest(BaseModel):
    alias: str = Field(..., min_length=1)
    model_path: str = Field(..., min_length=1)


class EvaluationRequest(BaseModel):
    model_name: str | None = None
    custom_model: str | None = None
    dataset_names: list[str] = Field(default_factory=list)
    selected_metrics: list[str] = Field(default_factory=list)
    tp_size_choice: str | int | None = "1"
    tp_size_input: str | int | None = None


class JobRecord(BaseModel):
    id: str
    status: JobStatus
    created_at: str
    updated_at: str
    request: EvaluationRequest
    result: dict[str, Any] | None = None
    error: str | None = None


app = FastAPI(title="FalcoEval API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=".*",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_executor = ThreadPoolExecutor(max_workers=1)
_jobs: dict[str, JobRecord] = {}
_jobs_lock = Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dataset_description(dataset_name: str) -> str:
    info = dataset_choices.get(dataset_name, {}).get("description", "")
    return str(info).strip() if info else "暂无数据集说明。"


def _available_metrics_for(dataset_names: list[str] | None) -> list[str]:
    metrics: list[str] = []
    for name in dataset_names or []:
        for metric in dataset_choices.get(name, {}).get("metrics", []):
            if metric not in metrics:
                metrics.append(metric)
    return metrics


def _combined_dataset_info(dataset_names: list[str] | None) -> str:
    names = dataset_names or []
    if not names:
        return "## 数据集概览\n请选择至少一个数据集。"

    lines = ["## 数据集概览"]
    for name in names:
        lines.append(_dataset_description(name))
    return "\n\n".join(lines)


def _model_options() -> list[dict[str, str]]:
    return [{"label": alias, "value": alias, "path": path} for alias, path in legacy_ui.REGISTERED_MODELS.items()]


def _dataset_options() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "data_type": DATA_TYPE_MAPPING.get(name),
            "metrics": dataset_choices.get(name, {}).get("metrics", []),
            "description": _dataset_description(name),
        }
        for name in legacy_ui.available_benchmarks()
    ]


def _update_job(job_id: str, **changes: Any) -> None:
    with _jobs_lock:
        job = _jobs[job_id]
        next_data = job.model_dump()
        next_data.update(changes)
        next_data["updated_at"] = _now()
        _jobs[job_id] = JobRecord(**next_data)


def _exception_message(exc: Exception) -> str:
    if gr is not None and isinstance(exc, gr.Error):
        return str(exc)
    return str(exc) or exc.__class__.__name__


def _run_evaluation(job_id: str, request: EvaluationRequest) -> None:
    _update_job(job_id, status="running")
    try:
        result = legacy_ui.evaluate(
            model_name=request.model_name or "",
            custom_model=request.custom_model,
            dataset_names=request.dataset_names,
            selected_metrics=request.selected_metrics,
            tp_size_choice=request.tp_size_choice,
            tp_size_input=request.tp_size_input,
            progress=None,
        )
    except Exception as exc:  # noqa: BLE001 - API should return task failures as job state.
        _update_job(job_id, status="failed", error=_exception_message(exc))
        return

    _update_job(job_id, status="succeeded", result=result, error=None)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/meta")
def meta() -> dict[str, Any]:
    default_datasets = legacy_ui.DEFAULT_DATASETS
    available_gpu_count = legacy_ui.available_gpu_count()
    gpu_choices = [str(i) for i in range(1, available_gpu_count + 1)] if available_gpu_count else ["1"]
    return {
        "models": _model_options(),
        "datasets": _dataset_options(),
        "default_datasets": default_datasets,
        "default_metrics": _available_metrics_for(default_datasets),
        "default_dataset_info": _combined_dataset_info(default_datasets),
        "available_gpu_count": available_gpu_count,
        "gpu_choices": gpu_choices,
        "default_gpu_choice": gpu_choices[0],
        "backend": "sglang",
    }


@app.post("/api/models/register")
def register_model(request: RegisterModelRequest) -> dict[str, Any]:
    alias = request.alias.strip()
    model_path = request.model_path.strip()
    if not alias:
        raise HTTPException(status_code=400, detail="请输入模型别名后再注册。")
    if not model_path:
        raise HTTPException(status_code=400, detail="请输入模型路径后再注册。")

    legacy_ui.REGISTERED_MODELS[alias] = model_path
    legacy_ui.save_registered_models()
    return {
        "status": f"已注册模型 `{alias}` -> `{model_path}`",
        "models": _model_options(),
        "selected": alias,
    }


@app.post("/api/datasets/selection")
def dataset_selection(dataset_names: list[str]) -> dict[str, Any]:
    return {
        "metrics": _available_metrics_for(dataset_names),
        "dataset_info": _combined_dataset_info(dataset_names),
    }


@app.post("/api/evaluations", response_model=JobRecord)
def start_evaluation(request: EvaluationRequest) -> JobRecord:
    if not request.dataset_names:
        raise HTTPException(status_code=400, detail="请至少选择一个数据集。")
    if not (request.custom_model or request.model_name):
        raise HTTPException(status_code=400, detail="请选择已注册模型，或输入一个临时模型路径。")

    unknown = [name for name in request.dataset_names if name not in legacy_ui.available_benchmarks()]
    if unknown:
        raise HTTPException(status_code=400, detail=f"不可用的数据集：{', '.join(unknown)}")

    job_id = uuid.uuid4().hex
    now = _now()
    record = JobRecord(
        id=job_id,
        status="queued",
        created_at=now,
        updated_at=now,
        request=request,
    )
    with _jobs_lock:
        _jobs[job_id] = record

    _executor.submit(_run_evaluation, job_id, request)
    return record


@app.get("/api/evaluations/{job_id}", response_model=JobRecord)
def get_evaluation(job_id: str) -> JobRecord:
    with _jobs_lock:
        record = _jobs.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Evaluation job not found.")
    return record
