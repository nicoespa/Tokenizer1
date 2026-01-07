from mini_tiktokenizer.regex_split import split_text
from mini_tiktokenizer.bpe import BPETokenizer
from mini_tiktokenizer.io import save_tokenizer, load_tokenizer

def train_small():
    corpus = (
        "Hola Nico!\n"
        "camión año acción\n"
        "Precio: $12.50\n"
        "emoji 🔥🚀 y símbolos ###\n"
    ) * 50
    return BPETokenizer.train(split_text(corpus), vocab_size=512)

def test_roundtrip_text():
    tok = train_small()
    samples = [
        "Hola, Nico!",
        "camión año acción",
        "Precio: $12.50",
        "🔥🚀 emojis",
        "líneas\ncon\nsaltos",
        "mix: Español + English + 123",
    ]
    for s in samples:
        ids = tok.encode_chunks(split_text(s))
        assert tok.decode(ids) == s

def test_save_load_roundtrip(tmp_path):
    tok = train_small()
    p = tmp_path / "tok.json"
    save_tokenizer(tok, str(p))
    tok2 = load_tokenizer(str(p))
    s = "Probando save/load 🔥"
    ids = tok2.encode_chunks(split_text(s))
    assert tok2.decode(ids) == s
