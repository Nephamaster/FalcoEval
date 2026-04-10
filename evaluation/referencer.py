from utils import DATA_TYPE_MAPPING
from typing import List
import json


def MCRef(data:dict):
    choices = data["choices"]
    answer = data['answer']
    ref=0
    for choice in choices:
        if answer == choice:
            break
        ref += 1
    return chr(ord('A')+ref)


def GenRef(data:dict):
    if 'answers' in data:
        answers = data['answers']
    else:
        answers = data['answer']
    if not isinstance(answers, list):
        ref = [answers]
    else:
        ref = answers
    return ref


def SinPef(data:dict):
    ref = data['answer']
    return ref


class Referencer:
    def __init__(self, dataset:str):
        self.dataset = dataset
        self.samples = self.load_dataset()
        self.data_type = DATA_TYPE_MAPPING.get(dataset, 'MultiChoice')
    
    def load_dataset(self) -> List[dict]:
        data_file = f'../datasets/{self.dataset}.jsonl'
        with open(data_file, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    
    def build_refernce(self):
        if self.data_type in ['MultiChoice']:
            refernces = [MCRef(sample) for sample in self.samples]
        elif self.data_type in ['Generation', 'Extraction','RelExtract','Precision']:
            if self.dataset != 'WikiEvent':
                refernces = [GenRef(sample) for sample in self.samples]
            else:
                pass
        elif self.data_type in ['Judgement', 'Math']:
            refernces = [SinPef(sample) for sample in self.samples]
        else:
            raise ValueError(f'Unsupportable dataset: {self.dataset}')
        return refernces