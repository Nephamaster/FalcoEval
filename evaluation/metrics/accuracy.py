from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch

class GenerateModel:
    def __init__(self, generator:str="moka-ai/m3e-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(generator, trust_remote_code=True)
        # self.eval_model = AutoModelForCausalLM.from_pretrained(
        #     generator, device_map="cuda:0", dtype=torch.float16, trust_remote_code=True)

    def get_tokens(self, sentence:str):
        """获取输入文本的token切分"""
        return self.tokenizer.tokenize(sentence)


def evaluate(predictions, references):
    """
    计算 Accuracy 指标
    - predictions: List[str/int]，模型输出的答案（如选项 A/B/C/D 或 True/False）
    - references: List[str/int]，真实标签
    """
    assert len(predictions) == len(references), "预测与参考答案数量不一致"

    correct = sum(p == r for p, r in zip(predictions, references))
    total = len(predictions)

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return round(accuracy, 2)


def token_efficiency(predictions, references):
    """
    计算 Token Efficiency 指标
    - predictions: List[str/int]，模型输出的答案（如选项 A/B/C/D 或 True/False）
    - references: List[str/int]，真实标签
    """
    assert len(predictions) == len(references), "预测与参考答案数量不一致"
    correct = sum(p == r for p, r in zip(predictions, references))
    total = len(predictions)

    accuracy = 100.0 * correct / total if total > 0 else 0.0

    tokencounter = GenerateModel('/mnt/disk4t/heyuxuan/data/models/FacebookAI/xlm-roberta-large')
    token_num = 0
    for p in predictions:
        tokens = tokencounter.get_tokens(str(p))
        token_num += len(tokens)
    avg_token_num = token_num / total
    print(f"avg_token_num: {avg_token_num}")
    te = accuracy / avg_token_num
    return {'Generation Length (tokens)':avg_token_num, 'Token Efficiency':te}