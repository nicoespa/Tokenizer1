# mini-tiktokenizer

Educational byte-level BPE tokenizer inspired by GPT-style tokenizers.

This project is **not** a production tokenizer.  
Its goal is to make tokenization *understandable* by implementing it
from scratch in a minimal and readable way.

---

## Features

- Byte-level BPE tokenization
- Regex-based pre-tokenization
- Train / encode / decode workflow
- Deterministic token IDs
- Simple CLI interface

---

## Install (development)

```bash
pip install -e .

## Project structure

```text
src/mini_tiktokenizer/
├── bpe.py           # BPE training logic
├── regex_split.py   # Pre-tokenization
├── tokenizer.py     # Core tokenizer
├── io.py            # Serialization helpers
└── cli.py           # CLI entrypoint
