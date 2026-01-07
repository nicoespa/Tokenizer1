import regex as re

GPT2_pattern = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\pL+| ?\pN+| ?[^\s\pL\pN]+|\s+(?!\S)|\s+"
)

def split_text(text:str) -> list[str]:
    return GPT2_pattern.findall(text)

