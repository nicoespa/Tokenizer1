# mini-tiktokenizer (educational)

Educational **byte-level BPE tokenizer** inspired by GPT-style tokenizers.

This repository contains a **from-scratch implementation** of a tokenizer similar in spirit to those used in modern Large Language Models (LLMs).  
The goal is not production performance, but **understanding**: making tokenization transparent, readable, and reproducible.

🔗 **Interactive playground (live demo):**  
https://tokenizer-playground-i2pegs22j-nicoespas-projects.vercel.app/

---

## Overview

Language models do not process words or characters — they process **tokens**.

Tokenization directly affects:
- prompt interpretation
- context window limits
- inference cost
- dataset preparation
- evaluation and debugging of LLM behavior

This project implements the **full tokenizer pipeline**, exposing every step that is usually hidden behind optimized libraries.

---

## What this project includes

- Byte-level text processing (Unicode-safe)
- Regex-based pre-tokenization
- Byte Pair Encoding (BPE) training
- Deterministic token ID assignment
- Encode / decode functionality
- JSON-based tokenizer serialization
- Command-line interface (CLI)
- Automated roundtrip tests
- A separate interactive web playground for visualization

---

## Installation (development)

    pip install -e .

---

## Running tests

    python -m pytest -q

---

## Command-line usage

### Train a tokenizer

    mini-tiktokenizer train \
      --text corpus.txt \
      --vocab-size 2048 \
      --out tok.json

### Encode text

    mini-tiktokenizer encode \
      --tokenizer tok.json \
      --text "Hola 🔥"

### Decode token IDs

    mini-tiktokenizer decode \
      --tokenizer tok.json \
      --ids "12,98,450,7"

---

## How it works (step-by-step)

### 1. Byte-level representation

All input text is converted to **bytes** before tokenization.  
This ensures deterministic behavior and full Unicode support, regardless of language or symbols.

---

### 2. Regex pre-tokenization

Before applying BPE, the text is split into chunks using regex rules.  
This step defines boundaries (words, spaces, punctuation) and improves merge quality.

**File:** `src/mini_tiktokenizer/regex_split.py`  
**Responsibility:** split raw text into pre-tokenized pieces

---

### 3. BPE training

Byte Pair Encoding is trained by:
1. Counting adjacent byte-pair frequencies
2. Merging the most frequent pairs
3. Repeating until the target vocabulary size is reached

**File:** `src/mini_tiktokenizer/bpe.py`  
**Responsibility:** learn merge rules and build the vocabulary

---

### 4. Tokenizer core

The core tokenizer applies:
- regex splitting
- byte conversion
- BPE merges
- token-to-id mapping

Encoding and decoding are fully deterministic.

**File:** `src/mini_tiktokenizer/tokenizer.py`  
**Responsibility:** encode text → token IDs, decode token IDs → text

---

### 5. Serialization

Tokenizer artifacts (vocabulary, merges, configuration) are saved and loaded as JSON files so the same tokenizer can be reused.

**File:** `src/mini_tiktokenizer/io.py`  
**Responsibility:** load/save tokenizer state

---

### 6. Command-line interface (CLI)

A minimal CLI exposes the full workflow:
- training
- encoding
- decoding

**File:** `src/mini_tiktokenizer/cli.py`  
**Responsibility:** user-facing interface for the tokenizer

---

## Project structure

    src/mini_tiktokenizer/
    ├── __init__.py
    ├── regex_split.py   # regex-based pre-tokenization
    ├── bpe.py           # BPE training and merge logic
    ├── tokenizer.py     # core encode/decode engine
    ├── io.py            # serialization helpers
    └── cli.py           # CLI entrypoint

    tests/
    └── test_roundtrip.py

---

## Notes

- This is an **educational implementation**, not intended for production use.
- For production systems, optimized libraries such as `tiktoken` should be used.
- This repository is designed as a reference to understand what production tokenizers abstract away.

