try:
    from .description import *
except ImportError:
    from description import *


# 数据集类型映射表
DATA_TYPE_MAPPING = {
    "ARC-Challenge": "MultiChoice",
    "ARC-Easy": "MultiChoice",
    "ARC-Easy_demo": "MultiChoice",
    "CommonsenseQA": "MultiChoice",
    "MMLU": "MultiChoice",
    "HellaSwag": "MultiChoice",
    "StrategyQA": "Judgement",
    "StrategyQA_demo": "Judgement",
    "TriviaQA-web": "Extraction",
    "TriviaQA-wiki": "Extraction",
    "COVID-QA": "Extraction",
    "HotpotQA": "Extraction",
    "HotpotQA_demo": "Extraction",
    "WikiEvent": "Extraction",
    "NaturalQuestions": "Extraction",
    "2WikiMultihopQA": "Generation",
    "2WikiMultihopQA_demo": "Generation",
    "Bamboogle": "Generation",
    "GPQA": "Generation",
    "GPQA_demo": "Generation",
    "LEXam": "Generation",
    "LEXam_demo": "Generation",
    "MedQA": "Generation",
    "MedQA_demo": "Generation",
    "MuSiQue": "Generation",
    "MuSiQue_demo": "Generation",
    "Qasper": "Generation",
    "NarrativeQA": "Generation",
    "PopQA": "Generation",
    "PopQA_demo": "Generation",
    "T-REx": "Precision",
    "T-REx_demo": "Precision",
    "Amazon": "Sentiment",
    "WoW": "Dialogue",
    "CamRest": "Dialogue",
    "TOXICSPANS": "Precision",
    "HalluQA": "Hallucination",
    "AIME25": "Math",
    "AIME25_demo": "Math",
    "AMC23": "Math",
    "AMC23_demo": "Math",
    "GSM8K": "Math",
    "GSM8K_demo": "Math",
    "MATH-500": "Math",
    "MATH-500_demo": "Math",
}


# --------------------- 数据集配置 -----------------------
dataset_choices = {
    "2WikiMultihopQA": {
        "metrics": ["EM", "F1", "TokenEfficiency"],
        "description": WikiMultihopQA
    },
    "2WikiMultihopQA_demo": {
        "metrics": ["EM", "F1", "TokenEfficiency"],
        "description": WikiMultihopQA
    },
    "AIME25": {
        "metrics": ["Accuracy", "TokenEfficiency"],
        "description": AIME
    },
    "AMC23": {
        "metrics": ["Accuracy", "TokenEfficiency"],
        "description": AMC
    },
    "ARC-Challenge": {
        "metrics": ["Accuracy", "TokenEfficiency"],
        "description": ARC
    },
    "ARC-Easy": {
        "metrics": ["Accuracy", "TokenEfficiency"],
        "description": ARC
    },
    "ARC-Easy_demo": {
        "metrics": ["Accuracy", "TokenEfficiency"],
        "description": ARC
    },
    "Bamboogle": {
        "metrics": ["EM", "F1", "TokenEfficiency"],
        "description": Bamboogle
    },
    # "CamRest": {
    #     "metrics": ["BLEU"],
    #     "description": CamRest
    # },
    "CommonsenseQA": {
        "metrics": ["Accuracy", "TokenEfficiency"],
        "description": CommonsenseQA
    },
    "COVID-QA": {
        "metrics": ["EM", "F1", "TokenEfficiency"],
        "description": COVID_QA
    },
    "GSM8K": {
        "metrics": ["Accuracy", "TokenEfficiency"],
        "description": GSM8K
    },
    "GSM8K_demo": {
        "metrics": ["Accuracy", "TokenEfficiency"],
        "description": GSM8K
    },
    # "HalluQA": {
    #     "metrics": ["EM", "F1"],
    #     "description": ARC
    # },
    "HellaSwag": {
        "metrics": ["Accuracy", "TokenEfficiency"],
        "description": HellaSwag
    },
    "HotpotQA": {
        "path": "/home/yaoguohui/workshop/ralm/spider/OUTPUT_DIR/BM25_test_qa/results.json",
        "metrics": ["EM", "F1", "TokenEfficiency"],
        "description": HotpotQA
    },
    "HotpotQA_demo": {
        "path": "/home/yaoguohui/workshop/ralm/spider/OUTPUT_DIR/BM25_test_qa/results.json",
        "metrics": ["EM", "F1", "TokenEfficiency"],
        "description": HotpotQA
    },
    "LEXam": {
        "metrics": ["BLEU", "ROUGE", "TokenEfficiency"],
        "description": LEXam
    },
    "MATH-500": {
        "metrics": ["Accuracy", "TokenEfficiency"],
        "description": MATH_500
    },
    "MATH-500_demo": {
        "metrics": ["Accuracy", "TokenEfficiency"],
        "description": MATH_500
    },
    "MedQA": {
        "metrics": ["EM", "F1", "TokenEfficiency"],
        "description": MedQA
    },
    "MMLU": {
        "metrics": ["Accuracy", "TokenEfficiency"],
        "description": MMLU
    },
    "MuSiQue": {
        "metrics": ["EM", "F1", "TokenEfficiency"],
        "description": MuSiQue
    },
    "NarrativeQA": {
        "metrics": ["BLEU", "ROUGE", "TokenEfficiency"],
        "description": NarrativeQA
    },
    # "NaturalQuestions": {
    #     "path": "/datasets/nq.json",
    #     "metrics": ["EM", "ROUGE"],
    #     "description": ARC
    # },
    "PopQA": {
        "metrics": ["BLEU", "EM", "F1", "TokenEfficiency"],
        "description": PopQA
    },
    "PopQA_demo": {
        "metrics": ["BLEU", "EM", "F1", "TokenEfficiency"],
        "description": PopQA
    },
    "Qasper": {
        "metrics": ["BLEU", "ROUGE", "TokenEfficiency"],
        "description": Qasper
    },
    "StrategyQA": {
        "metrics": ["Accuracy", "TokenEfficiency"],
        "description": StrategyQA
    },
    "TriviaQA": {
        "path": "/datasets/triviaqa.json",
        "metrics": ["EM", "F1", "BLEU", "TokenEfficiency"],
        "description": TriviaQA
    },
    "T-REx": {
        "metrics": ["Precision", "TokenEfficiency"],
        "description": T_REx
    },
    "T-REx_demo": {
        "metrics": ["Precision", "TokenEfficiency"],
        "description": T_REx
    },
    # "TOXICSPANS": {
    #     "metrics": ["Precision"],
    #     "description": ARC
    # },
    "WikiEvent": {
        "metrics": ["F1", "TokenEfficiency"],
        "description": WikiEvent
    },
    # "上传自定义数据集": {
    #     "path": None,
    #     "metrics": ["EM", "F1", "BLEU", "ROUGE"]
    # }
}

# ------------------ 模型路径 -------------------
model_paths = {
    "Qwen3-4B": "/mnt/disk4t/heyuxuan/data/models/Qwen/Qwen3-4B",
    "Llama-3.1-8B": "/mnt/disk4t/heyuxuan/data/models/meta-llama/Llama-3.1-8B"
}
