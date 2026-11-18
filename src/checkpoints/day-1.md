# ADAM-GPT

A from-scratch GPT-style language-model research project using PyTorch.

## Day 1: Environment, data and tokenizer

### Requirements
- Python 3.12
- uv
- Internet connection for the initial dataset download

### Setup

    uv sync

### Download Tiny Shakespeare

    mkdir -p data/raw
    curl -fL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o data/raw/tinyshakespeare.txt

Review the upstream repository's license and notices before
redistributing the corpus.

### Process the dataset

    uv run python -m adam.dataset

### Test

    uv run pytest -v

The pipeline creates train.bin, val.bin, vocab.json and
metadata.json under data/processed.

The split is sequential: 90% training and 10% validation.
Character IDs are stored as uint16 binary arrays.
The raw dataset's SHA-256 hash is recorded in metadata.json.

Preserve the raw corpus and its hash to reproduce the exact
dataset snapshot in future experiments.
EOF