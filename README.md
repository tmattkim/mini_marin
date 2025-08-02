# 🧠 Mini-Marin – Reproducible Tiny LLM Pipeline

This project implements a lightweight GPT-style language model in JAX/Flax for training, evaluation, and text generation on word-level tokenized data.

---

## 🤖 Model Features

- Word token inputs (vocab with `<unk>`, `<pad>`)  
- Transformer encoder with LayerNorm  
- Beam search + top-k sampling  
- Model saving/loading via `pickle`  
- JAX/Flax training on CPU/GPU/TPU  
- CLI interface for all operations  

---

## 🚀 Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Requirements include:
- `flax`
- `jax[cpu]`
- `datasets`
- `numpy`
- `wandb`

Notes:
- If you have a GPU/TPU, you might want to install a GPU-enabled JAX version from the official JAX install page instead of jax[cpu].
- The versions given are recent stable ones as of mid-2025.
- Update with pip install -U -r requirements.txt to get the latest minor versions.

---

## 📁 Project Structure

```
mini-marin/
├── data/
│   └── prepare_data.py         # Word-level tokenization, vocab build/load
├── model/
│   └── gpt_flax_model.py       # GPT-like model implemented in Flax
├── train.py                    # Training script (JAX/Flax)
├── generate.py                 # Text generation script (sampling & beam search)
├── utils.py                    # Loss functions, sampling helpers, beam search
├── eval.py                     # Evaluation script for test loss
├── checkpoints/                # Model and vocab checkpoints saved here
├── wandb/                      # WandB logging outputs
├── requirements.txt
└── README.md
```

---

## 🏋️‍♀️ Training
Train the model with vocabulary building:
```bash
python train.py \
  --config_path config.json \
  --train_data_path data/train.txt \
  --output_dir checkpoints \
  --build_vocab \
  --log_every 10
```

Notes:
- `--build_vocab` builds and saves `vocab.json` alongside the model.
- `config.json` should specify model and training hyperparameters (e.g., vocab size, block size, layers).
- Training logs loss per step and syncs with WandB if configured.

---

## ✅ Evaluation

Evaluate the trained model on a test set:

```bash
python eval.py \
  --config_path config.json \
  --eval_data_path data/test.txt \
  --model_path checkpoints/model.pkl \
  --wandb_project flax-gpt \
  --wandb_run_name eval-run
```

---

## ✍️ Generation

Generate text continuations using your trained model:

```bash
python generate.py \
  --config_path config.json \
  --model_path checkpoints/model.pkl \
  --vocab_path checkpoints/vocab.json \
  --prompt "Hello world" \
  --max_tokens 20 \
  --temperature 0.9 \
  --top_k 50
```

Streaming generation (outputs word by word):

```bash
python generate.py \
  --config_path config.json \
  --model_path checkpoints/model.pkl \
  --vocab_path checkpoints/vocab.json \
  --prompt "Hello world" \
  --max_tokens 20 \
  --temperature 0.9 \
  --top_k 50 \
  --stream
```

Beam search generation:
```bash
python generate.py \
  --config_path config.json \
  --model_path checkpoints/model.pkl \
  --vocab_path checkpoints/vocab.json \
  --prompt "Hello world" \
  --max_tokens 20 \
  --beam_search \
  --num_beams 4
```
---
## ⚙️ Command-Line Arguments Summary
| Argument            | Description                                     |
| ------------------- | ----------------------------------------------- |
| `--config_path`     | Path to JSON config file (default: config.json) |
| `--train_data_path` | Training text file (one sentence per line)      |
| `--eval_data_path`  | Evaluation text file                            |
| `--model_path`      | Path to saved model `.pkl` file                 |
| `--vocab_path`      | Path to `vocab.json` file                       |
| `--prompt`          | Prompt text for generation                      |
| `--max_tokens`      | Number of tokens to generate                    |
| `--temperature`     | Sampling temperature                            |
| `--top_k`           | Top-K sampling cutoff                           |
| `--stream`          | Stream output word-by-word                      |
| `--beam_search`     | Use beam search decoding                        |
| `--num_beams`       | Number of beams for beam search                 |
| `--build_vocab`     | Build vocab from training data                  |

---

## 📊 WandB Integration

Set your API key:

```bash
export WANDB_API_KEY=your_key_here
```
Training and evaluation metrics, as well as generation runs, are logged and visualized automatically.

---

## 🧪 Quick Test Pipeline

1. Prepare a simple training text file (train_simple.txt), one sentence per line.
2. Run training with vocab build flag.
3. Generate text with the trained model using the commands above.

---

## ✏️ Create a Tiny LLM

We will use the example `config.json` and `train_simple.text` files to train your own tiny LLM.

The `config.json` file is a JSON file that defines the model hyperparameters and config options we will use. Feel free to edit this and experiment!

```bash
{
  "vocab_size": 26,
  "block_size": 8,
  "n_layer": 2,
  "n_head": 2,
  "n_embd": 64,
  "seed": 42,
  "num_epochs": 200
}
```

Train the model by building the vocabulary from `train_simple.text` and training the model on the data:
```bash
python train.py --config_path config.json --train_data_path train_simple.txt --output_dir checkpoints --build_vocab --log_every 1
```

Generate text from the trained model. Run generation with your prompt, pointing to the trained model and vocab:
```bash
python generate.py \
  --config_path config.json \
  --model_path checkpoints/model.pkl \
  --vocab_path checkpoints/vocab.json \
  --prompt "hello" \
  --max_tokens 10 \
  --temperature 0.9 \
  --top_k 50
```
This will generate a 10-token continuation of `"hello"` using your trained model.

Optionally, you can enable stream generation for each token to appear one at a time:
```bash
python generate.py \
  --config_path config.json \
  --model_path checkpoints/model.pkl \
  --vocab_path checkpoints/vocab.json \
  --prompt "hello" \
  --max_tokens 10 \
  --temperature 0.9 \
  --top_k 50 \
  --stream
```

Enjoy creating LLMs from scratch!

---

## 📄 Credits

This project was inspired by Stanford's [Marin project](https://marin.community/), an open lab for building foundation models. I aimed to build a complete, transparent pipeline for training a small LLM from scratch.

---

## 📬 Questions?

Open an issue or contact [tmattkim@stanford.edu].

---
