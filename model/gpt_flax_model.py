import flax.linen as nn
import jax.numpy as jnp
import jax
import pickle
import os

class GPTConfig:
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd,
                 seed=0, num_epochs=10, vocab=None, inv_vocab=None):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.seed = seed
        self.num_epochs = num_epochs
        self.vocab = vocab or {}
        self.inv_vocab = inv_vocab or {}

class FlaxGPT(nn.Module):
    config: GPTConfig

    def setup(self):
        self.embed = nn.Embed(num_embeddings=self.config.vocab_size, features=self.config.n_embd)
        self.ln = nn.LayerNorm()
        self.head = nn.Dense(self.config.vocab_size)

    def __call__(self, x):
        x = self.embed(x)
        x = self.ln(x)
        logits = self.head(x)
        return logits

    def save_pretrained(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
            pickle.dump(self.params, f)

    def load_pretrained(self, model_path):
        with open(model_path, "rb") as f:
            params = pickle.load(f)
        self.params = params
