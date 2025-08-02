import os
import json
import argparse
import wandb
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from model.gpt_flax_model import GPTConfig, FlaxGPT
from data.prepare_data import load_and_tokenize_dataset, save_vocab
from utils import cross_entropy_loss, create_learning_rate_fn

def create_train_state(rng, config: GPTConfig, model, learning_rate_fn):
    params = model.init(rng, jnp.ones((1, config.block_size), dtype=jnp.int32))["params"]
    tx = optax.adamw(learning_rate=learning_rate_fn)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def train(args):
    with open(args.config_path, "r") as f:
        config_dict = json.load(f)

    # Build vocab if needed or load
    vocab = None
    inv_vocab = None
    if args.build_vocab:
        tokenized, vocab, inv_vocab = load_and_tokenize_dataset(args.train_data_path, config_dict["block_size"], None, True, args.vocab_size)
        save_vocab(vocab, os.path.join(args.output_dir, "vocab.json"))
        config_dict["vocab"] = vocab
        config_dict["inv_vocab"] = inv_vocab
    else:
        from data.prepare_data import load_vocab
        vocab, inv_vocab = load_vocab(os.path.join(args.output_dir, "vocab.json"))
        tokenized, _, _ = load_and_tokenize_dataset(args.train_data_path, config_dict["block_size"], vocab, False)

    config = GPTConfig(**config_dict)

    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=config_dict)

    model = FlaxGPT(config)
    learning_rate_fn = create_learning_rate_fn(config)

    rng = jax.random.PRNGKey(config.seed)
    state = create_train_state(rng, config, model, learning_rate_fn)

    for epoch in range(1, config.num_epochs + 1):
        for step, batch in enumerate(tokenized):
            batch = jnp.array(batch)
            def loss_fn(params):
                logits = model.apply({"params": params}, batch[:-1][None, :])
                loss = cross_entropy_loss(logits, batch[1:][None, :])
                return loss

            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)

            if step % args.log_every == 0:
                print(f"Epoch {epoch} Step {step}: loss = {loss:.4f}")
                wandb.log({"loss": float(loss), "epoch": epoch, "step": step})

    model.params = state.params
    model.save_pretrained(args.output_dir)
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config.json")
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--wandb_project", type=str, default="flax-gpt")
    parser.add_argument("--wandb_run_name", type=str, default="gpt-run")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--build_vocab", action="store_true")
    parser.add_argument("--vocab_size", type=int, default=10000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
