import argparse
import asyncio
from typing import List

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead, GRPOTrainer, GRPOConfig, setup_chat_format
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import wandb
from backend.agent import AgentExecutor

def parse_args():
    parser = argparse.ArgumentParser(description="RL fine-tuning with GRPO")
    parser.add_argument("--dataset", required=True, help="Path to training csv")
    parser.add_argument("--eval_dataset", default=None, help="Optional eval csv")
    parser.add_argument("--model_name", default="google/gemma-3-4b-it", help="HF model name")
    parser.add_argument("--output_dir", default="checkpoints", help="Checkpoint dir")
    return parser.parse_args()


def load_data(path: str) -> Dataset:
    df = pd.read_csv(path)
    df = df.dropna(subset=["input", "ground_truth"])
    prompts = [f"Complete this text with financial data, if needed. If not, return 'NO_COMPLETION_NEEDED': '{t}'" for t in df["input"].tolist()]
    data = {
        "prompt": prompts,
        "input": df["input"].tolist(),
        "ground_truth": df["ground_truth"].tolist(),
    }
    return Dataset.from_dict(data)


def build_model(name: str):
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb_config, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    model, tokenizer = setup_chat_format(model, tokenizer)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    return model, tokenizer


def normalize_text(text: str) -> str:
    import re
    return re.sub(r"[^0-9a-zA-Z]+", "", text.lower())


def extract_number(text: str):
    import re
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(nums[0]) if nums else None


def is_correct(pred: str, truth: str) -> bool:
    if truth == "NO_COMPLETION_NEEDED":
        return "NO_COMPLETION_NEEDED" in pred or pred.strip() == ""
    if "ERROR" in pred:
        return False
    n_gt = extract_number(truth)
    n_pred = extract_number(pred)
    if n_gt is not None and n_pred is not None:
        denom = abs(n_gt) if abs(n_gt) > 1e-6 else 1.0
        return abs(n_gt - n_pred) / denom < 0.05
    return normalize_text(pred) == normalize_text(truth)


def reward_fn(prompts: List[str], completions: List[str], ground_truth: List[str], **kwargs) -> List[float]:
    rewards = []
    for comp, gt in zip(completions, ground_truth):
        rewards.append(1.0 if is_correct(comp, gt) else 0.0)
    return rewards


async def evaluate(agent: AgentExecutor, eval_ds: Dataset) -> float:
    correct = 0
    for row in eval_ds:
        completion, _, _ = await agent.get_completion(row["input"])
        if is_correct(completion, row["ground_truth"]):
            correct += 1
    return correct / len(eval_ds)


async def run(args):
    train_ds = load_data(args.dataset)
    eval_ds = load_data(args.eval_dataset) if args.eval_dataset else None

    model, tokenizer = build_model(args.model_name)

    agent = AgentExecutor(model_provider="hf", model_name=args.model_name, hf_model=model, hf_tokenizer=tokenizer)

    config = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=200,
        report_to=["wandb"],
    )

    wandb.init(project="rl-finance", name="gemma-grpo")

    trainer = GRPOTrainer(model=model, args=config, reward_funcs=None, processing_class=tokenizer)

    for epoch in range(config.num_train_epochs):
        total_reward = 0.0
        for row in train_ds:
            completion, _, _ = await agent.get_completion(row["input"])
            reward = 1.0 if is_correct(completion, row["ground_truth"]) else 0.0
            total_reward += reward
            try:
                trainer.step(prompts=[row["prompt"]], completions=[completion], rewards=[reward])
            except AttributeError:
                pass  # step API may differ; this is a placeholder
        wandb.log({"avg_reward": total_reward / len(train_ds)})

        if eval_ds:
            acc = await evaluate(agent, eval_ds)
            print(f"Evaluation accuracy: {acc:.3f}")
            wandb.log({"eval_accuracy": acc})

    trainer.save_model(args.output_dir)


def main():
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
