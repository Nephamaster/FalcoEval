import json
import pandas as pd


# data = pd.read_parquet('datasets/AMC23/test-00000-of-00001.parquet')
# data = pd.read_parquet('datasets/bamboogle/test-00000-of-00001.parquet')
# with open('datasets/MuSiQue/musique_ans_v1.0_dev.jsonl', 'r') as f:
#     data = [json.loads(line) for line in f]
with open('datasets/LEXam.json', 'r') as f:
    data_list = json.load(f)

# print(data)
# data_list = []

# --- jsonl
# for d in data:
#     context = ''
#     para = d['paragraphs']
#     for p in para:
#         if p['is_supporting']:
#             context += p['paragraph_text'] + '\n'
#     answers = [d['answer']] + d['answer_aliases']
#     data_list.append(
#         {'context':context,'question': d['question'], 'answers': answers}
#     )

# --- parquet
# for i, row in data.iterrows():
    # print(row['context']['title'].tolist())
    # print(row)
    # if i > 10: 
    # break
    # sentences = row['context']['sentences']
    # context = ''
    # for sens in sentences:
        # context += ' '.join(sens.tolist()) + '\n'
    # data_list.append(
    #     {'context':context,'question': row['question'], 'answer': row['answer']}
    # )
    # data_list.append(
    #     {'question': row['question'], 'answer': row['golden_answers'].tolist()}
    # )

with open('datasets/LEXam.jsonl', 'w', encoding='utf-8') as f:
    for d in data_list:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')