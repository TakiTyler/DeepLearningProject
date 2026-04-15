import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer

from constants import *
from demo import load_csv
from preprocessing.parse import parse_ingredients, parse_steps
from preprocessing.clean import clean_ingredients, clean_steps, is_valid_recipe

# install the above with: pip install -U torch transformers peft trl bitsandbytes datasets pandas
# python.exe -m pip install --upgrade pip

def prepare_dataset(csv_path: str, num_samples: int = 50000):
    # """Loads, cleans, and formats the dataset for Hugging Face."""
    """Loads, cleans, and formats the dataset for Hugging Face.

    Args:
        csv_path (str): Path to the recipe dataset.
        num_samples (int, optional): Number of samples to load. Defaults to 50000 based on project proposal.

    Returns:
        Dataset: The filtered and formatted dataset.
    """
    print("--- loading and preprocessing dataset ---")
    df = load_csv(csv_path)

    formatted_data = {"text": []}
    valid_count = 0

    for _, row in df.iterrows():
        if valid_count >= num_samples:
            break

        ingredients = parse_ingredients(row['ingredients'])
        steps = parse_steps(row['steps'])

        ingredients = clean_ingredients(ingredients)
        steps = clean_steps(steps)

        if not is_valid_recipe(ingredients, steps):
            continue

        # convert lists to strings for the prompt
        ingr_str = ", ".join(ingredients)
        steps_str = " ".join([f"{i+1}. {step.capitalize()}" for i, step in enumerate(steps)])
        recipe_name = str(row['name']).title()

        # gemma chat template formatting
        prompt = (
            f"<start_of_turn>user\n"
            f"I have the following ingredients: {ingr_str}. What recipe can I make with these? Provide the name and steps.\n<end_of_turn>\n"
            f"<start_of_turn>model\n"
            f"**Recipe Name:** {recipe_name}\n"
            f"**Steps:**\n{steps_str}\n<end_of_turn>"
        )

        formatted_data["text"].append(prompt)
        valid_count += 1

    print(f"Prepared {len(formatted_data['text'])} valid recipes.")
    return Dataset.from_dict(formatted_data)

def main():
    rank = 4 # change to modify the rank of QLoRA

    ### prepare data ###
    dataset = prepare_dataset(RAW_RECI, num_samples=50000)  # 50k subset to match compute plan
    dataset = dataset.train_test_split(test_size=0.05)      # split into train and validation sets

    ### setup quantization ###
    model_id = "google/gemma-2b-it"

    # 4-bit precision
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    print("--- loading model and tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = 'right'

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )

    ### setup LoRA ###
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=rank,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],    # targeting attention blocks
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    ### training args ###
    # uses a batch size of 4, but waits to update weights until 4 sets of 4 recipes have been seen
    # basically giving a batch size of 16
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=10,
        num_train_epochs=1,             # try 1 for now
        fp16=True,
    )

    ### train ###
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("--- starting training ---")
    trainer.train()

    trainer.save_model(f"./gemma-2b-recipe-adapter-r{rank}")  # save the model so we don't re-train
    print("!!! training complete and model saved !!!")

if __name__ == "__main__":
    main()
