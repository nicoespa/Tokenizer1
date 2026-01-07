# src/mini_tiktokenizer/bpe.py
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
from typing import Dict, List, Tuple

def get_pairs(ids: List[int]) -> Counter[Tuple[int, int]]:
    pairs = Counter()
    for a, b in zip(ids, ids[1:]):
        pairs[(a, b)] += 1
    return pairs

def merge_pair(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    a, b = pair
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out

@dataclass
class BPETokenizer:
    # token_id -> bytes
    id_to_bytes: Dict[int, bytes]
    # bytes -> token_id
    bytes_to_id: Dict[bytes, int]
    # merge ranks: (id_a, id_b) -> rank (lower = earlier merge)
    merges: Dict[Tuple[int, int], int]

    @staticmethod
    def train(text_chunks: List[str], vocab_size: int = 2048) -> "BPETokenizer":
        assert vocab_size >= 256, "vocab_size must be >= 256"

        # init vocab with single bytes
        id_to_bytes = {i: bytes([i]) for i in range(256)}
        bytes_to_id = {v: k for k, v in id_to_bytes.items()}
        merges: Dict[Tuple[int, int], int] = {}

        # corpus as list of token-id sequences (each chunk separately)
        corpus_ids: List[List[int]] = []
        for ch in text_chunks:
            b = ch.encode("utf-8")
            corpus_ids.append(list(b))  # each byte is initial token id 0..255

        next_id = 256
        while next_id < vocab_size:
            # count pairs across all sequences
            pair_counts = Counter()
            for ids in corpus_ids:
                pair_counts.update(get_pairs(ids))

            if not pair_counts:
                break

            best_pair, _ = pair_counts.most_common(1)[0]

            # create merged token bytes
            new_bytes = id_to_bytes[best_pair[0]] + id_to_bytes[best_pair[1]]
            id_to_bytes[next_id] = new_bytes
            bytes_to_id[new_bytes] = next_id
            merges[best_pair] = len(merges)  # rank

            # apply merge to corpus
            corpus_ids = [merge_pair(ids, best_pair, next_id) for ids in corpus_ids]

            next_id += 1

        return BPETokenizer(id_to_bytes=id_to_bytes, bytes_to_id=bytes_to_id, merges=merges)

    def _encode_bytes_bpe(self, b: bytes) -> List[int]:
        # start from bytes -> ids
        ids = list(b)

        # repeatedly merge the best-ranked pair present
        while True:
            pairs = list(zip(ids, ids[1:]))
            # find mergeable pairs and pick the lowest-rank (highest priority)
            candidates = [(self.merges[p], p) for p in pairs if p in self.merges]
            if not candidates:
                break
            _, best_pair = min(candidates, key=lambda x: x[0])
            new_id = self.bytes_to_id[self.id_to_bytes[best_pair[0]] + self.id_to_bytes[best_pair[1]]]
            ids = merge_pair(ids, best_pair, new_id)

        return ids

    def encode_chunks(self, chunks: List[str]) -> List[int]:
        out: List[int] = []
        for ch in chunks:
            out.extend(self._encode_bytes_bpe(ch.encode("utf-8")))
        return out

    def decode(self, token_ids: List[int]) -> str:
        b = b"".join(self.id_to_bytes[t] for t in token_ids)
        return b.decode("utf-8", errors="replace")
