"""Adapter loading, generation, and interactive terminal demo.

Exposes `build_prompt` which is the single source of truth for how
ingredient lists become Gemma chat-template prompts — `finetune.py` and
every evaluator imports from here so training and inference never drift.
"""
import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = "google/gemma-2b-it"


def build_prompt(ingredients, target_name=None, target_steps=None):
    """Build the Gemma chat-template prompt.

    If target_name / target_steps are given, returns a FULL training example
    (user turn + model turn). Otherwise returns only the user turn, ready for
    generation — the model completes from `<start_of_turn>model\n`.
    """
    ingr_str = ", ".join(ingredients)
    user_turn = (
        f"<start_of_turn>user\n"
        f"I have the following ingredients: {ingr_str}. "
        f"What recipe can I make with these? Provide the name and steps.\n"
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    if target_name is None:
        return user_turn

    steps_str = " ".join(
        f"{i+1}. {step.capitalize()}" for i, step in enumerate(target_steps)
    )
    return (
        user_turn
        + f"**Recipe Name:** {target_name}\n"
        + f"**Steps:**\n{steps_str}\n<end_of_turn>"
    )


def format_reference(name, steps):
    """The ground-truth string the BLEU evaluator compares against."""
    steps_str = " ".join(
        f"{i+1}. {step.capitalize()}" for i, step in enumerate(steps)
    )
    return f"**Recipe Name:** {name}\n**Steps:**\n{steps_str}"


def _quant_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )


def load_base():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=_quant_config(), device_map="auto"
    )
    model.eval()
    return model, tokenizer


def load_adapter(rank, output_dir="."):
    """Load the base 4-bit Gemma and attach the LoRA adapter for the given rank."""
    model, tokenizer = load_base()
    adapter_path = os.path.join(output_dir, f"gemma-2b-recipe-adapter-r{rank}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def generate_recipe(model, tokenizer, ingredients, max_new_tokens=300):
    prompt = build_prompt(ingredients)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    # strip the prompt tokens; keep only the model's completion
    generated = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    # cut at end-of-turn if the model emits it
    if "<end_of_turn>" in text:
        text = text.split("<end_of_turn>")[0]
    return text.strip()


def make_generate_fn(model, tokenizer):
    """Return the standard `(ingredients) -> text` callback used by evaluate.py."""
    def gen(ingredients):
        return generate_recipe(model, tokenizer, ingredients)
    return gen


def evaluate_adapter(rank, num_eval=200, output_dir="."):
    """Load adapter for `rank`, run BLEU on the held-out test split."""
    from evaluate import evaluate_model
    from splits import build_splits

    _, _, test = build_splits()
    model, tokenizer = load_adapter(rank, output_dir=output_dir)
    gen_fn = make_generate_fn(model, tokenizer)
    return evaluate_model(gen_fn, test, run_name=f"qlora_r{rank}",
                          rank=rank, num_eval=num_eval,
                          output_dir=output_dir)


def _interactive(rank, output_dir="."):
    print(f"Loading adapter r={rank}...")
    model, tokenizer = load_adapter(rank, output_dir=output_dir)
    print("Ready. Enter comma-separated ingredients (or 'quit').")
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line or line.lower() in {"quit", "exit", "q"}:
            break
        ingredients = [x.strip().lower() for x in line.split(",") if x.strip()]
        print()
        print(generate_recipe(model, tokenizer, ingredients))
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default=".", help="Directory where the adapter is saved.")
    args = parser.parse_args()
    _interactive(args.rank, output_dir=args.output_dir)
