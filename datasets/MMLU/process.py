import pandas as pd
import json
from pathlib import Path


dir = Path('test')
data_file = dir.glob('*.csv')
length = 0
with open(f'MMLU-test.jsonl', 'w') as f:
    for file in data_file:
        data = pd.read_csv(file,sep=',')
        for i in range(len(data)):
            question = data.iloc[i,0]
            choices = [str(data.iloc[i,1]),
                    str(data.iloc[i,2]),
                    str(data.iloc[i,3]),
                    str(data.iloc[i,4])]
            answer = choices[ord(data.iloc[i,5])-ord('A')]
            term = {
                'question':question,
                'choices':choices,
                'answer': answer
            }
            json_line = json.dumps(term,ensure_ascii=False)
            f.write(json_line+'\n')