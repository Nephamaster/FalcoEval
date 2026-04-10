import json

def preprocess_qasper(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(output_path, 'w', encoding='utf-8') as fout:
        for paper_id, paper in data.items():
            context = build_full_text(paper['full_text'])
            for qa in paper.get("qas", []):
                question = qa.get("question")

                # 收集所有形式的答案（free_form、extractive_spans、yes_no）
                answers = set()
                for ann in qa.get("answers", []):
                    ans = ann.get("answer", {})
                    # 处理自由文本答案
                    if ans.get("free_form_answer"):
                        answers.add(ans["free_form_answer"])
                    # 处理可抽取答案
                    for span in ans.get("extractive_spans", []):
                        answers.add(span)
                    # 处理 yes/no 类型
                    if ans.get("yes_no") in ["yes", "no"]:
                        answers.add(ans["yes_no"])

                output_item = {
                    "context": context,
                    "question": question,
                    "answers": list(answers)
                }
                fout.write(json.dumps(output_item, ensure_ascii=False) + "\n")

def build_full_text(sections):
    paragraphs = []
    for sec in sections:
        section_title = sec.get("section_name", "")
        section_texts = sec.get("paragraphs", [])
        paragraphs.append(section_title) if section_title is not None else paragraphs.append('')
        section_texts = [text for text in section_texts if text is not None]
        paragraphs.extend(section_texts)
    try:
        return "\n".join(paragraphs)
    except:
        print(paragraphs)

# 使用示例
preprocess_qasper("qasper-test-v0.3.json", "Qasper-test.jsonl")
