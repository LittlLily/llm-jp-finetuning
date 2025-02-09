from model import load_model
from datasets import load_dataset
from tqdm import tqdm

def evaluate():
    model, tokenizer = load_model()
    dataset = load_dataset("json", data_files="data/elyza-tasks-100-TV_0.jsonl")

    results = []
    for dt in tqdm(dataset["train"]):
        input_text = dt["input"]
        prompt = f"### 指示\n{input_text}\n### 回答\n"
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

        outputs = model.generate(**inputs, max_new_tokens=512, use_cache=True, do_sample=False, repetition_penalty=1.2)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).split('\n### 回答')[-1]

        results.append({"task_id": dt["task_id"], "input": input_text, "output": prediction})

    with open("outputs/evaluation_results.jsonl", "w", encoding="utf-8") as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")

if __name__ == "__main__":
    evaluate()
