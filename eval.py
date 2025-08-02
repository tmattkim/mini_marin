import json
import argparse
import jax.numpy as jnp
import wandb
from model.gpt_flax_model import GPTConfig, FlaxGPT
from data.prepare_data import load_dataset, load_vocab
from utils import cross_entropy_loss

def evaluate(args):
    with open(args.config_path, "r") as f:
        config_dict = json.load(f)

    vocab, inv_vocab = load_vocab(args.vocab_path)
    tokenized, _, _ = load_dataset(args.eval_data_path, config_dict["block_size"], vocab, False)

    config_dict["vocab"] = vocab
    config_dict["inv_vocab"] = inv_vocab
    config = GPTConfig(**config_dict)

    wandb.init(project=args.wandb_project, name=args.wandb_run_name + "-eval", config=config_dict)

    model = FlaxGPT(config)
    model.load_pretrained(args.model_path)

    total_loss = 0
    total_batches = 0
    for batch in tokenized:
        batch = jnp.array(batch)
        logits = model(batch[:-1][None, :])
        loss = cross_entropy_loss(logits, batch[1:][None, :])
        total_loss += float(loss)
        total_batches += 1

    avg_loss = total_loss / total_batches
    print(f"✅ Evaluation loss: {avg_loss:.4f}")
    wandb.log({"eval_loss": avg_loss})
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config.json")
    parser.add_argument("--eval_data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, default="flax-gpt")
    parser.add_argument("--wandb_run_name", type=str, default="gpt-run")
    args = parser.parse_args()

    evaluate(args)
