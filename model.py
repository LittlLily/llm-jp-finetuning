from unsloth import FastLanguageModel
import torch
from config import MODEL_ID

def load_model():
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        dtype=None,
        load_in_4bit=load_in_4bit,
        trust_remote_code=True,
    )

    return model, tokenizer
