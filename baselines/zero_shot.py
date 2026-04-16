"""Zero-shot Gemma 2B baseline — loads the un-finetuned instruct model
and runs BLEU on the same held-out test set used by the QLoRA runs.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from splits import build_splits
from inference import load_base, make_generate_fn
from evaluate import evaluate_model


def main():
    _, _, test = build_splits()
    model, tokenizer = load_base()
    gen_fn = make_generate_fn(model, tokenizer)
    evaluate_model(gen_fn, test, run_name="zero_shot", rank="-")


if __name__ == "__main__":
    main()
