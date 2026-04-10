import json
import argparse
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import DATA_TYPE_MAPPING
from promptor import Promptor
from referencer import Referencer


class GenerateModel:
    def __init__(self, generator:str="moka-ai/m3e-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(generator, trust_remote_code=True)

    def get_tokens(self, sentence:str):
        """获取输入文本的token切分"""
        return self.tokenizer.tokenize(sentence)


class Predictor:
    def __init__(self, model_path="meta-llama/Llama-2-7b-chat-hf", device="cuda"):
        print(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="cuda"
        )
        self.model.eval()
        self.device = device

    def generate(self, prompt: str, max_new_tokens=512) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                )
            ln_outputs = self.model(**inputs, output_hidden_states=False, return_dict=True)
            logits = ln_outputs.logits  # [1, seq_len, vocab_size]
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output_text[len(prompt):].strip()


class TaskProcessor:
    def __init__(self, model:Predictor, dataset:str):
        self.model = model
        self.dataset = dataset
        self.promptor = Promptor(dataset)
        self.referencer = Referencer(dataset)
        self.data_type = DATA_TYPE_MAPPING.get(dataset, 'MultiChoice')
        self.tokencounter = GenerateModel('FacebookAI/xlm-roberta-large')
    
    def parse_output(self, output):
        if self.data_type in ['MultiChoice']:
            for token in ["A", "B", "C", "D", "E", "F"]:
                if token in output:
                    return token
        elif output is None:
            return ''
        else:
            return output.strip()

    def predict(self, progressor=None):
        predictions = []
        latencies = []
        token_nums = []
        print('Building prompts...')
        prompts = self.promptor.build_prompt()
        for prompt in progressor.tqdm(prompts, desc="评估中"):
            start = time.perf_counter()
            output = self.model.generate(prompt)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
            pred = self.parse_output(output)
            print('model output:', pred)
            predictions.append(pred)
            token_nums.append(len(self.tokencounter.get_tokens(pred)))
        print('Building references...')
        references = self.referencer.build_refernce()
        return predictions, references, latencies, token_nums


def predict(dataset:str, model_path:str, task_type:str=None, output:str=None, progressor=None):
    data_type = task_type or DATA_TYPE_MAPPING.get(dataset)
    if data_type is None:
        raise ValueError(f"数据集类型未知，请通过 --data_type 指定")
    model = Predictor(model_path)
    processor = TaskProcessor(model, dataset)
    predictions, references, latencies, token_nums = processor.predict(progressor)
    output_data = {
        "predictions": predictions,
        "references": references,
        "latencies": latencies,
        "lengths": token_nums
    }
    output_path = output
    if output_path is None:
        output_path = f'../output/{dataset}_pred_ref.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"预测完成，结果已保存至: {output_path}")
    
    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="待测数据集")
    parser.add_argument("--output", type=str, default=None, help="输出预测文件路径")
    parser.add_argument("--data_type", type=str, default=None, help="数据集类型")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="待测模型路径")

    args = parser.parse_args()
    predict(args.dataset, args.model_path, args.data_type, args.output)

# CUDA_VISIBLE_DEVICES=0 python run_predict.py --dataset StrategyQA_short --model_path /mnt/disk4t/heyuxuan/models/meta-llama/Llama-2-7b-chat