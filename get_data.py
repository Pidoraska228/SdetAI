from datasets import load_dataset
import os

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

print("Загрузка датасета...")
# Using streaming to save disk space and RAM
# We'll use a very small subset for testing
dataset = load_dataset("bigcode/the-stack-v2-dedup", data_dir="data/java", split="train", streaming=True)

print("Сохраняем данные в data/java_samples.jsonl...")
with open("data/java_samples.jsonl", "w", encoding="utf-8") as f:
    count = 0
    for example in dataset:
        f.write(example['content'] + "\n")
        count += 1
        if count >= 100:  # Start small: 100 files
            break

print(f"Готово! Сохранено {count} примеров.")
