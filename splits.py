"""Deterministic train/val/test split shared by fine-tuning and evaluation.

Centralizing split logic here guarantees that `finetune.py`, the baselines,
and the evaluator all see the same held-out test set — preventing leakage.
"""
import random
from datasets import Dataset

from constants import RAW_RECI
from demo import load_csv
from preprocessing.parse import parse_ingredients, parse_steps
from preprocessing.clean import clean_ingredients, clean_steps, is_valid_recipe


def _iter_valid_recipes(csv_path, num_samples):
    df = load_csv(csv_path)
    count = 0
    for _, row in df.iterrows():
        if count >= num_samples:
            break
        ingredients = clean_ingredients(parse_ingredients(row['ingredients']))
        steps = clean_steps(parse_steps(row['steps']))
        if not is_valid_recipe(ingredients, steps):
            continue
        yield {
            "name": str(row['name']).title() if row['name'] else "Untitled",
            "ingredients": ingredients,
            "steps": steps,
        }
        count += 1


def build_splits(csv_path=RAW_RECI, num_samples=50000, seed=42):
    """Return (train, val, test) HF Datasets with ratios 90/5/5.

    Each example is a dict with keys: name, ingredients (list), steps (list).
    Downstream code is responsible for formatting into a training prompt.
    """
    records = list(_iter_valid_recipes(csv_path, num_samples))
    rng = random.Random(seed)
    rng.shuffle(records)

    n = len(records)
    n_train = int(0.90 * n)
    n_val = int(0.05 * n)

    train = records[:n_train]
    val = records[n_train:n_train + n_val]
    test = records[n_train + n_val:]

    def to_ds(rows):
        return Dataset.from_dict({
            "name": [r["name"] for r in rows],
            "ingredients": [r["ingredients"] for r in rows],
            "steps": [r["steps"] for r in rows],
        })

    return to_ds(train), to_ds(val), to_ds(test)


if __name__ == "__main__":
    train, val, test = build_splits()
    print(f"train: {len(train)}")
    print(f"val:   {len(val)}")
    print(f"test:  {len(test)}")
    print(f"example: {train[0]}")
