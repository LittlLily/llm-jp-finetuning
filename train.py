from model import load_model
from dataset import load_training_data
from utils import display_gpu_info
from transformers import TrainingArguments
from trl import SFTTrainer

def train():
    model, tokenizer = load_model()
    dataset = load_training_data()

    args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        logging_steps=10,
        warmup_steps=10,
        save_steps=100,
        save_total_limit=2,
        max_steps=-1,
        learning_rate=2e-4,
        output_dir="outputs",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        max_seq_length=512,
        dataset_text_field="formatted_text",
        packing=False,
        args=args,
    )

    display_gpu_info()
    trainer.train()

if __name__ == "__main__":
    train()
