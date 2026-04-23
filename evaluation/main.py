import multiprocessing
import json
import os
import socket
import time
from pathlib import Path

import gradio as gr
import torch

try:
    from .paths import dataset_path
    from .run_eval import eval
    from .run_predict import create_predictor, default_max_new_tokens_for_task, predict
    from .utils import DATA_TYPE_MAPPING, dataset_choices, model_paths
except ImportError:
    from paths import dataset_path
    from run_eval import eval
    from run_predict import create_predictor, default_max_new_tokens_for_task, predict
    from utils import DATA_TYPE_MAPPING, dataset_choices, model_paths


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

DEFAULT_SERVER_PORT = 7860
PORT_SCAN_LIMIT = 20
REGISTERED_MODELS_PATH = Path(__file__).resolve().parent / "registered_models.json"


SUPPORTED_UI_DATA_TYPES = {"MultiChoice", "Judgement", "Extraction", "Generation", "Math", "Precision"}
UNSUPPORTED_UI_DATASETS = {"WikiEvent", "2WikiMultihopQA", "2WikiMultihopQA_demo"}
REGISTERED_MODELS = dict(model_paths)
ACTIVE_PREDICTOR = None
ACTIVE_MODEL_PATH = None
ACTIVE_BACKEND = "sglang"
ACTIVE_TP_SIZE = 1


def load_registered_models() -> dict[str, str]:
    if not REGISTERED_MODELS_PATH.exists():
        return {}

    try:
        with REGISTERED_MODELS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(data, dict):
        return {}

    models = {}
    for alias, model_path in data.items():
        if isinstance(alias, str) and isinstance(model_path, str):
            alias = alias.strip()
            model_path = model_path.strip()
            if alias and model_path:
                models[alias] = model_path
    return models


def save_registered_models():
    user_models = {
        alias: path
        for alias, path in REGISTERED_MODELS.items()
        if model_paths.get(alias) != path
    }
    with REGISTERED_MODELS_PATH.open("w", encoding="utf-8") as f:
        json.dump(user_models, f, indent=2, ensure_ascii=False)


REGISTERED_MODELS.update(load_registered_models())


def available_gpu_count() -> int:
    try:
        count = int(torch.cuda.device_count())
    except Exception:
        count = 0
    return max(count, 0)


AVAILABLE_GPU_COUNT = available_gpu_count()


def _is_runnable_benchmark(dataset_name: str) -> bool:
    data_type = DATA_TYPE_MAPPING.get(dataset_name)
    if data_type not in SUPPORTED_UI_DATA_TYPES:
        return False
    if dataset_name in UNSUPPORTED_UI_DATASETS:
        return False
    if dataset_name not in dataset_choices:
        return False
    return dataset_path(dataset_name).exists()


def available_benchmarks() -> list[str]:
    return sorted(name for name in dataset_choices if _is_runnable_benchmark(name))


AVAILABLE_BENCHMARKS = available_benchmarks()
DEFAULT_DATASETS = AVAILABLE_BENCHMARKS[:1] if AVAILABLE_BENCHMARKS else []


def _port_is_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", port))
        except OSError:
            return False
    return True


def resolve_server_port() -> int:
    configured_port = os.getenv("GRADIO_SERVER_PORT")
    if configured_port:
        return int(configured_port)

    for port in range(DEFAULT_SERVER_PORT, DEFAULT_SERVER_PORT + PORT_SCAN_LIMIT):
        if _port_is_available(port):
            return port

    raise OSError(
        f"Cannot find an empty port in range {DEFAULT_SERVER_PORT}-{DEFAULT_SERVER_PORT + PORT_SCAN_LIMIT - 1}. "
        "Set GRADIO_SERVER_PORT to override the default scan range."
    )


def _dataset_description(dataset_name: str) -> str:
    info = dataset_choices.get(dataset_name, {}).get("description", "暂无数据集说明。")
    return str(info).strip() if info else "暂无数据集说明。"


def _combined_dataset_info(dataset_names: list[str] | None) -> str:
    names = dataset_names or []
    if not names:
        return "## 数据集概览\n请选择至少一个数据集。"

    lines = ["## 数据集概览"]
    for name in names:
        lines.append(_dataset_description(name))
    return "\n\n".join(lines)


def _available_metrics_for(dataset_names: list[str] | None) -> list[str]:
    names = dataset_names or []
    metrics = []
    for name in names:
        for metric in dataset_choices.get(name, {}).get("metrics", []):
            if metric not in metrics:
                metrics.append(metric)
    return metrics


def on_dataset_change(dataset_names: list[str] | None, current_metrics: list[str] | None):
    metrics = _available_metrics_for(dataset_names)
    selected_metrics = [metric for metric in (current_metrics or []) if metric in metrics]
    return (
        gr.update(choices=metrics, value=selected_metrics),
        gr.update(value=_combined_dataset_info(dataset_names)),
    )


def register_model(alias: str, model_path: str):
    alias = (alias or "").strip()
    model_path = (model_path or "").strip()

    if not alias:
        status = "请输入模型别名后再注册。"
    elif not model_path:
        status = "请输入模型路径后再注册。"
    else:
        REGISTERED_MODELS[alias] = model_path
        save_registered_models()
        status = f"已注册模型 `{alias}` -> `{model_path}`"

    choices = list(REGISTERED_MODELS.keys())
    selected = alias if alias in REGISTERED_MODELS else (choices[0] if choices else None)
    return (
        gr.update(choices=choices, value=selected),
        gr.update(value=""),
        gr.update(value=""),
        status,
    )


def _shutdown_active_predictor():
    global ACTIVE_PREDICTOR, ACTIVE_MODEL_PATH, ACTIVE_TP_SIZE
    if ACTIVE_PREDICTOR is not None:
        ACTIVE_PREDICTOR.shutdown()
    ACTIVE_PREDICTOR = None
    ACTIVE_MODEL_PATH = None
    ACTIVE_TP_SIZE = 1


def _resolve_tp_size(tp_size_choice, tp_size_input) -> int:
    value = tp_size_input if tp_size_input not in (None, "") else tp_size_choice
    if value in (None, ""):
        return 1
    tp_size = int(value)
    if tp_size < 1:
        raise gr.Error("显卡数量至少为 1。")
    if AVAILABLE_GPU_COUNT and tp_size > AVAILABLE_GPU_COUNT:
        raise gr.Error(f"显卡数量不能超过当前可见 GPU 数量：{AVAILABLE_GPU_COUNT}。")
    return tp_size


def get_or_create_session_predictor(model_path: str, backend: str = "sglang", tp_size: int = 1):
    global ACTIVE_PREDICTOR, ACTIVE_MODEL_PATH, ACTIVE_BACKEND, ACTIVE_TP_SIZE

    if (
        ACTIVE_PREDICTOR is not None
        and ACTIVE_MODEL_PATH == model_path
        and ACTIVE_BACKEND == backend
        and ACTIVE_TP_SIZE == tp_size
    ):
        ACTIVE_PREDICTOR._session_reused = True
        ACTIVE_PREDICTOR._session_load_ms = 0.0
        return ACTIVE_PREDICTOR, True, 0.0

    _shutdown_active_predictor()
    load_start = time.perf_counter()
    predictor = create_predictor(model_path=model_path, backend=backend, tp_size=tp_size)
    load_ms = (time.perf_counter() - load_start) * 1000
    predictor._session_reused = False
    predictor._session_load_ms = load_ms
    ACTIVE_PREDICTOR = predictor
    ACTIVE_MODEL_PATH = model_path
    ACTIVE_BACKEND = backend
    ACTIVE_TP_SIZE = tp_size
    return predictor, False, load_ms


def evaluate(
    model_name: str,
    custom_model: str | None,
    dataset_names: list[str] | None,
    selected_metrics: list[str] | None,
    tp_size_choice,
    tp_size_input,
    progress=gr.Progress(track_tqdm=True),
):
    names = dataset_names or []
    if not names:
        raise gr.Error("请至少选择一个数据集。")

    custom_model = (custom_model or "").strip()
    if custom_model:
        model_path = custom_model
        model_label = custom_model
    else:
        if not model_name or model_name not in REGISTERED_MODELS:
            raise gr.Error("请选择已注册模型，或输入一个临时模型路径。")
        model_path = REGISTERED_MODELS[model_name]
        model_label = model_name

    tp_size = _resolve_tp_size(tp_size_choice, tp_size_input)
    predictor, model_reused_for_run, session_load_ms = get_or_create_session_predictor(
        model_path,
        backend="sglang",
        tp_size=tp_size,
    )
    results = {
        "model": model_label,
        "model_path": model_path,
        "backend": "sglang",
        "tp_size": tp_size,
        "session": {
            "model_reused_for_run": model_reused_for_run,
            "cold_start_load_ms": round(float(session_load_ms), 2),
            "warmup_note": (
                "新模型首次运行时，SGLang 可能需要数分钟进行引擎预热。"
            ),
        },
        "datasets": {},
    }

    for dataset_name in names:
        supported_metrics = dataset_choices[dataset_name]["metrics"]
        metrics_to_use = [metric for metric in (selected_metrics or []) if metric in supported_metrics]
        if not metrics_to_use:
            metrics_to_use = supported_metrics
        data_type = DATA_TYPE_MAPPING[dataset_name]
        max_new_tokens = default_max_new_tokens_for_task(data_type)

        prednref = predict(
            dataset=dataset_name,
            model_path=model_path,
            progressor=progress,
            backend="sglang",
            predictor=predictor,
            max_new_tokens=max_new_tokens,
            tp_size=tp_size,
        )
        result = eval(dataset=dataset_name, prednref=prednref, metrics=metrics_to_use)
        results["datasets"][dataset_name] = {
            "data_type": data_type,
            "metrics_used": metrics_to_use,
            "generation": {
                "max_new_tokens": prednref["manifest"]["generation"]["max_new_tokens"],
            },
            "runtime": prednref.get("runtime", {}),
            "result": result,
        }

    return results


def build_demo():
    with gr.Blocks(css="""
    .app-title h1 {
        font-size: 2.5rem;
        margin-bottom: 0.35rem;
    }
    .app-subtitle p {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 0.75rem;
    }
    .status-bar {
        padding: 0.85rem 1rem;
        margin: 0.75rem 0 1.25rem 0;
        background: #f5f7fb;
        border: 1px solid #dbe4f0;
        border-radius: 10px;
        color: #1f2937;
        line-height: 1.6;
    }
    .dataset-panel h2 {
        font-size: 1.75rem;
        margin-bottom: 0.75rem;
    }
    .dataset-panel h3 {
        font-size: 1.35rem;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .dataset-panel h4,
    .dataset-panel h5,
    .dataset-panel h6 {
        margin-top: 0.75rem;
    }
    """) as demo:
        gr.Markdown("# FalcoEval", elem_classes=["app-title"])
        gr.Markdown(
            "面向大语言模型的统一自动化评测平台，支持多基准、多指标与可复用推理引擎。",
            elem_classes=["app-subtitle"],
        )
        gr.HTML(
            """
<div class="status-bar">
                <strong>当前状态</strong><br>
1. 首次加载新模型时，SGLang 可能需要数分钟完成引擎预热。<br>
2. 后续若继续使用同一模型，系统会复用已加载引擎，评测速度会明显更快。<br>
3. 页面仅展示已映射且可被当前评测流程正常运行的数据集。<br>
4. 当前服务可见 GPU 数量：""" + str(AVAILABLE_GPU_COUNT) + """
</div>
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                dataset_info = gr.Markdown(_combined_dataset_info(DEFAULT_DATASETS), elem_classes=["dataset-panel"])
            with gr.Column(scale=2):
                gr.Markdown("## 模型配置")

                model_selector = gr.Dropdown(
                    label="已注册模型",
                    choices=list(REGISTERED_MODELS.keys()),
                    value=(list(REGISTERED_MODELS.keys())[0] if REGISTERED_MODELS else None),
                    allow_custom_value=False,
                )

                custom_model_path = gr.Textbox(
                    label="临时模型路径",
                    placeholder="可选：为本次评测临时指定本地路径或 HuggingFace 模型路径",
                )

                gpu_choices = [str(i) for i in range(1, AVAILABLE_GPU_COUNT + 1)] if AVAILABLE_GPU_COUNT else ["1"]
                default_gpu_choice = gpu_choices[0]
                gr.Markdown("## 显卡配置")
                gr.Markdown(
                    f"当前服务可见 GPU 数量：**{AVAILABLE_GPU_COUNT}**。SGLang 会将该值作为张量并行数量（`tp_size`）使用。"
                )
                tp_size_choice = gr.Dropdown(
                    label="显卡数量（下拉选择）",
                    choices=gpu_choices,
                    value=default_gpu_choice,
                    allow_custom_value=False,
                )
                tp_size_input = gr.Textbox(
                    label="显卡数量（手动输入，可覆盖下拉选择）",
                    placeholder="可选：输入一个正整数，例如 1 / 2 / 4",
                )

                with gr.Accordion("注册模型", open=False):
                    register_alias = gr.Textbox(label="模型别名", placeholder="例如：Qwen2.5-7B-Instruct")
                    register_path = gr.Textbox(label="模型路径", placeholder="例如：Qwen/Qwen2.5-7B-Instruct")
                    register_btn = gr.Button("注册模型")
                    register_status = gr.Markdown()

                register_btn.click(
                    fn=register_model,
                    inputs=[register_alias, register_path],
                    outputs=[model_selector, register_alias, register_path, register_status],
                )

                gr.Markdown("## 数据集配置")
                dataset_selector = gr.Dropdown(
                    label="评测基准",
                    choices=AVAILABLE_BENCHMARKS,
                    value=DEFAULT_DATASETS,
                    multiselect=True,
                )

                metric_selector = gr.CheckboxGroup(
                    label="评测指标",
                    choices=_available_metrics_for(DEFAULT_DATASETS),
                    value=[],
                )

                dataset_selector.change(
                    fn=on_dataset_change,
                    inputs=[dataset_selector, metric_selector],
                    outputs=[metric_selector, dataset_info],
                )

                submit_btn = gr.Button("开始评测", variant="primary")
                output = gr.JSON(label="评测结果")

                submit_btn.click(
                    fn=evaluate,
                    inputs=[model_selector, custom_model_path, dataset_selector, metric_selector, tp_size_choice, tp_size_input],
                    outputs=output,
                )

                gr.Markdown(
                    """
### 使用说明
- 可以先注册模型别名与路径，后续直接在下拉框中选择使用。
- 也可以不注册，直接填写临时模型路径进行单次评测。
- 可以选择或手动输入显卡数量，系统会将其作为 SGLang 的 `tp_size` 使用。
- 当前页面仅展示已映射且能被当前评测流程正常运行的数据集。
- 选择多个数据集时，系统会逐个执行评测，并按数据集返回结果汇总。
                    """
                )
    return demo


if __name__ == "__main__":
    multiprocessing.freeze_support()
    demo = build_demo()
    demo.launch(share=False, server_name="0.0.0.0", server_port=resolve_server_port())
# /mnt/disk4t/heyuxuan/data/models/Qwen/Qwen3-8B-AWQ
