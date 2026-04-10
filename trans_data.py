import json


with open('datasets/MATH-500.jsonl', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

data_list = []
for d in data:
    data_list.append(
        {'question': d['problem'], 'answer': d['answer']}
    )

with open('datasets/MATH-500.jsonl', 'w', encoding='utf-8') as f:
    for d in data_list:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')

with open('datasets/MATH-500_short.jsonl', 'w', encoding='utf-8') as f:
    for d in data_list[:20]:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')