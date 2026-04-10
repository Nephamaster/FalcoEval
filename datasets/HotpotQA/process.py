import json

def preprocess_hotpotqa(input_path: str, output_path: str):
    """
    将 HotpotQA 多跳问答任务预处理为标准 QA 格式
    每条记录格式：
        {
            "question": ...,
            "context": ...,
            "answer": ...
        }
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed = []
    for item in data:
        context_blocks = []
        for title, sentences in item["context"]:
            block = f"{title}:\n" + " ".join(sentences)
            context_blocks.append(block)
        full_context = "\n\n".join(context_blocks)

        processed_item = {
            "context": full_context,
            "question": item["question"],
            "answer": item["answer"]
        }
        processed.append(processed_item)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in processed:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

    print(f"✅ 预处理完成，已保存到：{output_path}")


# 示例调用
if __name__ == "__main__":
    preprocess_hotpotqa("hotpot_dev_distractor_v1.json", "HotpotQA-devlopment.jsonl")
