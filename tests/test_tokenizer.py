from adam import CharTokenizer


def test_round_trip():
    text = "Hello, ADAM!\n"
    tokenizer = CharTokenizer.from_text(text)
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_save_and_load(tmp_path):
    text = "abc cab"
    tokenizer = CharTokenizer.from_text(text)

    path = tmp_path / "vocab.json"
    tokenizer.save(path)
    loaded = CharTokenizer.load(path)

    assert loaded.decode(loaded.encode(text)) == text
    assert loaded.vocab_size == tokenizer.vocab_size


def test_unknown_character():
    tokenizer = CharTokenizer.from_text("abc")

    try:
        tokenizer.encode("z")
    except ValueError:
        pass
    else:
        assert False, "Expected unknown character to raise ValueError"
