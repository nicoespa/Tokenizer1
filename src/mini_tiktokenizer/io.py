import json
from typing import Dict, Tuple
from .bpe import BPETokenizer

def save_tokenizer(tok: BPETokenizer, path: str, pattern_name: str = "gpt2") -> None:
    payload = {
        "version": 1,
        "pattern": pattern_name,
        "id_to_bytes": {str(i): tok.id_to_bytes[i].hex() for i in tok.id_to_bytes},
        "merges": {f"{a},{b}": r for (a, b), r in tok.merges.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

def load_tokenizer(path: str) -> BPETokenizer:
    payload = json.load(open(path, "r", encoding="utf-8"))

    id_to_bytes: Dict[int, bytes] = {int(k): bytes.fromhex(v) for k, v in payload["id_to_bytes"].items()}
    bytes_to_id = {v: k for k, v in id_to_bytes.items()}

    merges: Dict[Tuple[int, int], int] = {}
    for k, r in payload["merges"].items():
        a, b = k.split(",")
        merges[(int(a), int(b))] = int(r)

    return BPETokenizer(id_to_bytes=id_to_bytes, bytes_to_id=bytes_to_id, merges=merges)
