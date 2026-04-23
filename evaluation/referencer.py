from typing import List

try:
    from .data_loader import load_jsonl_dataset
    from .paths import dataset_path
    from .utils import DATA_TYPE_MAPPING
except ImportError:
    from data_loader import load_jsonl_dataset
    from paths import dataset_path
    from utils import DATA_TYPE_MAPPING


def MCRef(data:dict):
    choices = [str(choice).strip() for choice in data["choices"]]
    answer = str(data['answer']).strip()
    if len(answer) == 1 and answer.upper() in "ABCDEF":
        return answer.upper()
    ref = 0
    for choice in choices:
        if answer == choice:
            break
        ref += 1
    if ref >= len(choices):
        raise ValueError(f"Answer is not in choices: {answer}")
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
    ref = str(data['answer']).strip()
    return ref


class Referencer:
    def __init__(self, dataset:str):
        self.dataset = dataset
        self.data_type = DATA_TYPE_MAPPING.get(dataset, 'MultiChoice')
        self.samples = self.load_dataset()
    
    def load_dataset(self) -> List[dict]:
        data_file = dataset_path(self.dataset)
        return load_jsonl_dataset(data_file, self.data_type)
    
    def build_refernce(self):
        if self.data_type in ['MultiChoice']:
            refernces = [MCRef(sample) for sample in self.samples]
        elif self.data_type in ['Generation', 'Extraction','RelExtract','Precision']:
            if self.dataset != 'WikiEvent':
                refernces = [GenRef(sample) for sample in self.samples]
            else:
                raise NotImplementedError("WikiEvent reference building is not implemented.")
        elif self.data_type in ['Judgement', 'Math']:
            refernces = [SinPef(sample) for sample in self.samples]
        else:
            raise ValueError(f'Unsupportable dataset: {self.dataset}')
        return refernces
