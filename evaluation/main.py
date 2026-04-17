import torch
import json
import gradio as gr
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from run_predict import predict
from run_eval import eval
from utils import *


os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


# ------------------ 模型管理 ------------------
current_model = None
current_tokenizer = None
current_model_path = None

def load_model_safely(path):
    global current_model, current_tokenizer, current_model_path
    if current_model and current_model_path != path:
        del current_model
        del current_tokenizer
        torch.cuda.empty_cache()
        print(f"❌ 卸载旧模型：{current_model_path}")
    if current_model_path != path:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path).to("cuda").eval()
        current_model = model
        current_tokenizer = tokenizer
        current_model_path = path
        print(f"✅ 加载新模型：{path}")
    return current_model, current_tokenizer


# ------------------ 通用数据加载 ------------------
def load_dataset(path_or_file):
    if isinstance(path_or_file, str):
        with open(path_or_file, "r", encoding="utf-8") as f:
            return json.load(f)
    elif hasattr(path_or_file, "name"):
        return json.load(path_or_file)
    else:
        raise ValueError("无效的数据源")


# ------------------ 每个数据集加载器 ------------------
def load_hotpotqa(path_or_file): return load_dataset(path_or_file)
def load_triviaqa(path_or_file): return load_dataset(path_or_file)
def load_nq(path_or_file): return load_dataset(path_or_file)
def load_custom(file): return load_dataset(file)


# ------------------ 数据集切换时更新 UI ------------------
def on_dataset_change(dataset_name):
    metrics = dataset_choices[dataset_name]["metrics"]
    upload_visible = dataset_name == "上传自定义数据集"
    return gr.update(choices=metrics, value=[]), gr.update(visible=upload_visible), gr.update(value=dataset_choices[dataset_name].get("description", "暂无说明"))


def evaluate(model_name='Qwen/Qwen2.5-3B',
             custom_model=None,
             dataset_name='ARC-Easy_demo',
             selected_metrics=['Accuracy'],
             file=None,
             progress=gr.Progress(track_tqdm=True)):
    model_path = custom_model if custom_model is not None else model_paths[model_name]
    prednref = predict(dataset=dataset_name, model_path=model_path, progressor=progress, backend="sglang")
    result = eval(dataset=dataset_name, prednref=prednref, metrics=selected_metrics)
    print('Done')
    return result


# ------------------ Gradio UI ------------------
with gr.Blocks() as demo:
    with gr.Column():
        gr.Markdown("# 大模型性能自动化评估系统")
        with gr.Row():
            with gr.Column(scale=1) as right_column:
                gr.Markdown("## 数据集说明")
                dataset_info = gr.Markdown(dataset_choices["ARC-Easy_demo"].get("description", "暂无说明"))
            with gr.Column(scale=2) as left_column:
                # 模型选择部分
                gr.Markdown("## 模型配置")
                
                # 添加单选按钮来选择模型类型
                model_type = gr.Radio(
                    choices=["预定义模型", "自定义模型"],
                    label="模型类型",
                    value="预定义模型",
                    interactive=True
                )
                
                # 预定义模型选择器
                model_selector = gr.Dropdown(
                    label="选择预定义模型", 
                    choices=list(model_paths.keys()), 
                    value="LLaMA2-7B-chat",
                    visible=True
                )
                
                # 自定义模型路径输入
                custom_model_path = gr.Textbox(
                    label="本地模型路径",
                    placeholder="请输入本地模型的完整路径，例如：/path/to/your/model",
                    visible=False
                )
                
                # 模型类型切换逻辑
                def on_model_type_change(model_type_value):
                    if model_type_value == "预定义模型":
                        return gr.update(visible=True), gr.update(visible=False)
                    else:
                        return gr.update(visible=False), gr.update(visible=True)
                
                model_type.change(
                    fn=on_model_type_change,
                    inputs=model_type,
                    outputs=[model_selector, custom_model_path]
                )
                
                # 数据集选择部分
                gr.Markdown("## 数据集配置")
                dataset_selector = gr.Dropdown(
                    label="选择数据集", 
                    choices=list(dataset_choices.keys()), 
                    value="ARC-Easy_demo"
                )

                metric_selector = gr.CheckboxGroup(
                    label="选择评估指标", 
                    choices=dataset_choices["ARC-Easy_demo"]["metrics"]
                )
                file_input = gr.File(
                    label="上传自定义数据集（JSON 格式）", 
                    visible=False
                )

                dataset_selector.change(
                    fn=on_dataset_change, 
                    inputs=dataset_selector, 
                    outputs=[metric_selector, file_input, dataset_info]
                )

                # 评估按钮
                submit_btn = gr.Button("开始评估", variant="primary")
                output = gr.JSON(label="评估结果")

                submit_btn.click(
                    fn=evaluate,
                    inputs=[model_selector, custom_model_path, dataset_selector, metric_selector, file_input],
                    outputs=output
                )
                
                # 添加使用说明
                gr.Markdown("""
                ### 💡 使用说明
                - **预定义模型**：从下拉列表中选择已配置的模型
                - **自定义模型**：输入本地模型的完整路径（支持HuggingFace格式的模型）
                - 模型路径示例：`/home/user/models/llama-7b` 或 `./my-finetuned-model`
                """)

demo.launch(share=True, server_name='0.0.0.0', server_port=1234)
