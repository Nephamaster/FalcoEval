import re


CHOICE_PATTERN = re.compile(
    r"(?:^|\b)(?:answer|option|choice)?\s*[:：]?\s*[\(\[]?\s*([A-F])\s*[\)\]]?(?:\b|$)",
    re.IGNORECASE,
)


def extract_multichoice(output: str) -> str:
    if not output:
        return ""
    match = CHOICE_PATTERN.search(output.strip())
    return match.group(1).upper() if match else output.strip()[:1].upper()


def extract_judgement(output: str) -> str:
    if not output:
        return ""
    text = output.strip().lower()
    if re.search(r"\b(yes|true)\b", text):
        return "yes"
    if re.search(r"\b(no|false)\b", text):
        return "no"
    return output.strip()


def extract_math(output: str) -> str:
    if not output:
        return ""

    boxed = re.findall(r"\\boxed\{([^{}]+)\}", output)
    if boxed:
        return boxed[-1].strip()

    numbers = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?|[-+]?\d+\s*/\s*\d+", output)
    if numbers:
        return numbers[-1].replace(",", "").replace(" ", "")

    return output.strip()


def extract_answer(output: str, data_type: str) -> str:
    if output is None:
        return ""
    if data_type == "MultiChoice":
        return extract_multichoice(output)
    if data_type == "Judgement":
        return extract_judgement(output)
    if data_type == "Math":
        return extract_math(output)
    return output.strip()
