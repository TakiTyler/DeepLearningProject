import argparse
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer

from splits import build_splits
from inference import build_prompt

# install deps with: pip install -U torch transformers peft trl bitsandbytes datasets pandas scikit-learn sacrebleu


def _format_for_sft(ds):
    """Turn a split from `splits.build_splits` into a {'text': [...]} SFT dataset."""
    texts = [
        build_prompt(ex["ingredients"], target_name=ex["name"], target_steps=ex["steps"])
        for ex in ds
    ]
    return Dataset.from_dict({"text": texts})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=4, help="QLoRA rank (ablation: 4, 16, 64)")
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    rank = args.rank

    ### prepare data ###
    train_raw, val_raw, _test_raw = build_splits(num_samples=args.num_samples)
    train_ds = _format_for_sft(train_raw)
    val_ds = _format_for_sft(val_raw)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    ### setup quantization ###
    model_id = "google/gemma-2b-it"

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
        device_map="auto",
    )

    ### setup LoRA ###
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=rank,
        lora_alpha=2 * rank,  # standard heuristic so alpha scales with rank
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    ### training args ###
    training_args = TrainingArguments(
        output_dir=f"./results/checkpoints-r{rank}",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=10,
        num_train_epochs=args.epochs,
        fp16=True,
    )

    ### train ###
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
    )

    print(f"--- starting training (rank={rank}) ---")
    trainer.train()

    trainer.save_model(f"./gemma-2b-recipe-adapter-r{rank}")
    print(f"!!! training complete: ./gemma-2b-recipe-adapter-r{rank} !!!")


if __name__ == "__main__":
    main()
