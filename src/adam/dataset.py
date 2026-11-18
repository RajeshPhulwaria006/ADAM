import hashlib
import json
import numpy as np

from pathlib import Path
from adam import CharTokenizer

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "src/data/raw_data/tinyshakespeare.txt"
OUT = ROOT / "src/data/processed"

if not RAW.exists():
    raise FileNotFoundError(
        f"Dataset not found: {RAW}\n"
        "Download Tiny Shakespeare first."
    )

raw_bytes = RAW.read_bytes()
text = raw_bytes.decode("utf-8")
text = text.replace("\r\n", "\n").replace("\r", "\n")
text = text.lstrip("\ufeff")

if len(text) < 2:
    raise ValueError("Dataset is too small.")

# Sequential split: first 90% for training, last 10% for validation.
split = int(len(text) * 0.90)
train_text = text[:split]
val_text = text[split:]

# Shared vocabulary allows validation characters to be encoded.
tokenizer = CharTokenizer.from_text(text)

if tokenizer.vocab_size > 65536:
    raise ValueError("Vocabulary too large for uint16 IDs.")

train_ids = np.asarray(
    tokenizer.encode(train_text), dtype=np.uint16
)
val_ids = np.asarray(
    tokenizer.encode(val_text), dtype=np.uint16
)

OUT.mkdir(parents=True, exist_ok=True)

train_ids.tofile(OUT / "train.bin")
val_ids.tofile(OUT / "val.bin")
tokenizer.save(OUT / "vocab.json")

metadata = {
    "source_url": (
        "https://raw.githubusercontent.com/karpathy/"
        "char-rnn/master/data/tinyshakespeare/input.txt"
    ),
    "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
    "total_characters": len(text),
    "train_characters": len(train_text),
    "validation_characters": len(val_text),
    "vocab_size": tokenizer.vocab_size,
    "train_fraction": 0.90,
    "split_method": "sequential",
    "encoding": "uint16 token IDs",
}

(OUT / "metadata.json").write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)

print(json.dumps(metadata, indent=2))
print(f"\nProcessed files saved to: {OUT}")
