from datasets import load_dataset
import os

# Создаем папку для данных
os.makedirs("data", exist_ok=True)

print("Загрузка открытого датасета (tiny_shakespeare)...")
# Используем tiny_shakespeare, так как он не требует авторизации
dataset = load_dataset("tiny_shakespeare", split="train", streaming=True)

print("Сохраняем данные...")
with open("data/java_samples.jsonl", "w", encoding="utf-8") as f:
    for i, example in enumerate(dataset):
        # В этом датасете данные в поле 'text'
        f.write(example['text'] + "\n")
        if i >= 1000: break 

print("Готово! Файл data/java_samples.jsonl создан.")
