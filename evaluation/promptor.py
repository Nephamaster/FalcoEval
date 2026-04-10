from utils import DATA_TYPE_MAPPING
from typing import List
import json


def MCPrompt(data:dict):
    context = data.get('context','')
    q = data["question"]
    choices = data["choices"]
    choice_str = "\n".join([f"{chr(ord('A')+index)}. {text}" for index, text in enumerate(choices)])
    prompt = ''
    if context != '':
        prompt += f'Given the context\n"{context}",\n'
    prompt += f"read the question or text below, and choose the most appropriate option:\n{q}\noptions:\n{choice_str}\nRespond only to the option you think is right, and do not output anything else.\n"
    return prompt


def SingleQAPrompt(data:dict):
    context = data.get('context','')
    q = data["question"]
    prompt = ''
    if context != '':
        prompt += f'Given the context\n"{context}",\n'
    prompt += f"answer the question below.\nQuestion: {q}\nAnswer:\n"
    return prompt


def MathPrompt(data:dict):
    q = data["question"]
    prompt = f"Answer the math problem below.\nQuestion: {q}\Respond only to the final answer(the value) you've calculated:\n"
    return prompt


class Promptor:
    def __init__(self, dataset:str):
        self.dataset = dataset
        self.samples = self.load_dataset()
        self.data_type = DATA_TYPE_MAPPING.get(dataset, 'MultiChoice')
    
    def load_dataset(self) -> List[dict]:
        data_file = f'../datasets/{self.dataset}.jsonl'
        with open(data_file, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    
    def build_prompt(self):
        if self.data_type == 'MultiChoice':
            prompts = [MCPrompt(sample) for sample in self.samples]
        elif self.data_type in ['Judgement', 'Generation']:
            prompts = [SingleQAPrompt(sample) for sample in self.samples]
        elif self.data_type in ['Extraction','RelExtract','Precision']:
            if self.dataset != 'WikiEvent':
                prompts = [SingleQAPrompt(sample) for sample in self.samples]
            else:
                pass
        elif self.data_type == 'Math':
            prompts = [MathPrompt(sample) for sample in self.samples]
        else:
            raise ValueError(f'Unsupportable dataset: {self.dataset}')
        return prompts