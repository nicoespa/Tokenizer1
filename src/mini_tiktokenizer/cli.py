# src/mini_tiktokenizer/cli.py
import json
import argparse
from .regex_split import split_text
from .bpe import BPETokenizer

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--text", required=True, help="path to training text file")
    t.add_argument("--vocab_size", type=int, default=2048)
    t.add_argument("--out", default="tokenizer.json")

    e = sub.add_parser("encode")
    e.add_argument("--tokenizer", default="tokenizer.json")
    e.add_argument("--text", required=True)

    d = sub.add_parser("decode")
    d.add_argument("--tokenizer", default="tokenizer.json")
    d.add_argument("--ids", required=True, help="comma-separated ids")

    args = p.parse_args()

    if args.cmd == "train":
        raw = open(args.text, "r", encoding="utf-8").read()
        chunks = split_text(raw)
        tok = BPETokenizer.train(chunks, vocab_size=args.vocab_size)

        payload = {
            "id_to_bytes": {str(k): tok.id_to_bytes[k].hex() for k in tok.id_to_bytes},
            "merges": {f"{a},{b}": r for (a, b), r in tok.merges.items()},
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        print(f"saved -> {args.out}")

    else:
        payload = json.load(open(args.tokenizer, "r", encoding="utf-8"))
        id_to_bytes = {int(k): bytes.fromhex(v) for k, v in payload["id_to_bytes"].items()}
        bytes_to_id = {v: k for k, v in id_to_bytes.items()}
        merges = {}
        for k, r in payload["merges"].items():
            a, b = k.split(",")
            merges[(int(a), int(b))] = int(r)
        tok = BPETokenizer(id_to_bytes=id_to_bytes, bytes_to_id=bytes_to_id, merges=merges)

        if args.cmd == "encode":
            ids = tok.encode_chunks(split_text(args.text))
            print(ids)

        if args.cmd == "decode":
            ids = [int(x) for x in args.ids.split(",") if x.strip()]
            print(tok.decode(ids))

if __name__ == "__main__":
    main()
