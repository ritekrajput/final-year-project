from transformers import AutoTokenizer

MODEL_NAME = "roberta-base"
MAX_LEN = 256

_tokenizer = None
_tokenizer_error = None


def get_tokenizer():
    global _tokenizer, _tokenizer_error

    if _tokenizer is not None:
        return _tokenizer
    if _tokenizer_error is not None:
        return None

    try:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        return _tokenizer
    except Exception as exc:
        _tokenizer_error = str(exc)
        return None


def preprocess_text(patient_text, relative_text, device):
    tokenizer = get_tokenizer()
    if tokenizer is None:
        return None

    text = patient_text + " </s> " + relative_text
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )

    return {
        "input_ids": encoding["input_ids"].to(device),
        "attention_mask": encoding["attention_mask"].to(device),
    }
