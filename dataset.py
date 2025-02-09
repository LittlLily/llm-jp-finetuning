from datasets import load_dataset

def load_training_data():
    dataset = load_dataset("json", data_files="data/fixed_ichikara-instruction.json")

    prompt_template = "### 指示\n{}\n### 回答\n{}"

    def format_prompt(examples):
        input_text = examples["text"]
        output_text = examples["output"]
        formatted_text = prompt_template.format(input_text, output_text)
        return {"formatted_text": formatted_text}

    dataset = dataset.map(format_prompt, num_proc=4)
    return dataset
