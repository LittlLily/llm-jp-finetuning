from model import load_model
from config import HF_TOKEN, NEW_MODEL_ID

def upload_model():
    model, tokenizer = load_model()
    model.push_to_hub_merged(NEW_MODEL_ID + "_lora", tokenizer=tokenizer, save_method="lora", token=HF_TOKEN, private=True)

if __name__ == "__main__":
    upload_model()
