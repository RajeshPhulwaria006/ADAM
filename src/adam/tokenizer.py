import json
from pathlib import Path


class CharTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for i, ch in enumerate(vocab)}

    @classmethod
    def from_text(cls, text):
        if not text:
            raise ValueError("Text cannot be empty.")
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, text):
        unknown = set(text) - self.stoi.keys()
        if unknown:
            raise ValueError(f"Unknown characters: {unknown}")
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        try:
            return "".join(self.itos[i] for i in ids)
        except KeyError as exc:
            raise ValueError(f"Invalid token ID: {exc.args[0]}") from exc

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.vocab, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path):
        vocab = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(vocab)
