from transformers import AutoTokenizer

MODEL_NAME = "roberta-base"
MAX_LEN = 256

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_text(patient_text, relative_text, device):
    text = patient_text + " </s> " + relative_text

    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    return {
        "input_ids": encoding["input_ids"].to(device),
        "attention_mask": encoding["attention_mask"].to(device)
    }
