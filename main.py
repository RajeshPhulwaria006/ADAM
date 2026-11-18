import numpy as np
from pathlib import Path
from src.adam import CharTokenizer

root = Path('src/data')
raw = (root / 'raw_data/tinyshakespeare.txt').read_text(encoding='utf-8')
tok = CharTokenizer.load(root / 'processed/vocab.json')

ids = tok.encode(raw)
decoded = tok.decode(ids)

print('Original characters:', len(raw))
print('Encoded tokens:', len(ids))
print('Vocabulary size:', tok.vocab_size)
print('Exact reconstruction:', decoded == raw)
print('First 100 characters:', repr(decoded[:100]))
