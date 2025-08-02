import argparse
import json
import jax
import jax.numpy as jnp
import numpy as np
from model.gpt_flax_model import GPTConfig, FlaxGPT
from utils import top_k_logits, beam_search_generate
import wandb

def sample(model, params, config, prompt, max_tokens, temperature=1.0, top_k=None, stream=False, rng=None):
    if rng is None:
        rng = jax.random.PRNGKey(0)  # fixed seed for reproducibility

    tokens = [config.vocab.get(w, config.vocab["<unk>"]) for w in prompt.strip().split()]
    input_ids = jnp.array(tokens)[None, :]

    for _ in range(max_tokens):
        logits = model.apply({"params": params}, input_ids)
        next_token_logits = logits[:, -1, :] / temperature

        if top_k is not None:
            next_token_logits = top_k_logits(next_token_logits, top_k)

        rng, subkey = jax.random.split(rng)
        next_token = jax.random.categorical(subkey, next_token_logits)
        input_ids = jnp.concatenate([input_ids, next_token[:, None]], axis=-1)

        if stream:
            word = config.inv_vocab.get(int(next_token[0]), "<unk>")
            print(word + " ", end="", flush=True)

    if not stream:
        decoded = " ".join([config.inv_vocab.get(int(tok), "<unk>") for tok in input_ids[0]])
        return decoded

def generate(args):
    with open(args.config_path, "r") as f:
        config_dict = json.load(f)

    vocab_path = args.vocab_path or "checkpoints/vocab.json"
    from data.prepare_data import load_vocab
    vocab, inv_vocab = load_vocab(vocab_path)
    config_dict["vocab"] = vocab
    config_dict["inv_vocab"] = inv_vocab

    config = GPTConfig(**config_dict)

    model = FlaxGPT(config)
    model.load_pretrained(args.model_path)

    wandb.init(project=args.wandb_project, name=args.wandb_run_name + "-gen", config=config_dict)

    if args.beam_search:
        tokens = [config.vocab.get(w, config.vocab["<unk>"]) for w in args.prompt.strip().split()]
        input_ids = jnp.array(tokens)[None, :]
        output_ids = beam_search_generate(
            model=model,
            params=model.params,
            input_ids=input_ids,
            max_length=args.max_tokens,
            num_beams=args.num_beams,
            eos_token_id=None,
            temperature=args.temperature,
            top_k=args.top_k
        )
        # Decode best beam (index 0) only
        best_sequence = output_ids[0, 0]
        decoded = " ".join([config.inv_vocab.get(int(tok), "<unk>") for tok in best_sequence])
        print("📝 Generated with beam search:")
        print(decoded)
    else:
        if args.stream:
            print("▶️ Streaming output:")
        rng = jax.random.PRNGKey(42)  # Fixed seed, could expose as an arg
        output = sample(
            model=model,
            params=model.params,
            config=config,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            stream=args.stream,
            rng=rng
        )
        if not args.stream:
            print("📝 Generated text:")
            print(output)

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config.json")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--beam_search", action="store_true")
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--wandb_project", type=str, default="flax-gpt")
    parser.add_argument("--wandb_run_name", type=str, default="gpt-run")
    args = parser.parse_args()

    generate(args)
