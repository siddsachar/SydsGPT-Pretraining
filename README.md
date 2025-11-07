# SydsGPT-Pretraining

A from-scratch, readable implementation of a GPT-style language model in PyTorch, plus a Jupyter notebook for pretraining and text generation experiments.

## Highlights

- Minimal, well-structured PyTorch implementation:
  - Multi-Head Self-Attention with causal masking
  - Pre-norm Transformer blocks with residual connections
  - Token + positional embeddings, LayerNorm, GELU feed-forward
  - Tie-able output projection head sized to vocab
- Tokenization via OpenAI's GPT-2 BPE (`tiktoken`)
- Simple greedy text generation utility (`modules/GenerateSimple.py`)
- Training and sampling walkthrough in `pretraining.ipynb`

## Repository structure

```
SydsGPT-Pretraining/
├─ pretraining.ipynb          # End-to-end training + generation walkthrough
├─ model/
│  └─ SydsGPT.py              # SydsGPT model definition (Transformer LM)
├─ modules/
│  ├─ DataLoader.py           # Tiny dataset + dataloader backed by tiktoken
│  ├─ GenerateSimple.py       # Greedy decode helper (argmax sampling)
│  ├─ MultiHeadAttention.py   # Causal MHA (with mask and dropout)
│  ├─ TransformerBlock.py     # Pre-norm block: MHA + FFN + residual
│  ├─ FeedForward.py          # GELU MLP (hidden dim = 4x embedding)
│  ├─ LayerNorm.py            # Simple LayerNorm (scale/shift)
│  └─ GELU.py                 # GELU activation
└─ .gitignore                 # Ignores caches, venvs, checkpoints, logs, data
```

## Model at a glance

- Class: `SydsGPT` in `model/SydsGPT.py`
- Forward: `(batch, seq_len) -> logits (batch, seq_len, vocab_size)`
- Core config keys (dictionary):
  - `vocab_size` (int): tokenizer vocabulary size
  - `embedding_dim` (int): model hidden size
  - `context_length` (int): max sequence length (positional embedding size)
  - `num_layers` (int): number of Transformer blocks
  - `num_heads` (int): attention heads (must divide `embedding_dim`)
  - `dropout` (float): dropout prob for attention/MLP
  - `qkv_bias` (bool): add bias to Q/K/V projections

## Requirements

This repo uses:
- `torch` (PyTorch)
- `tiktoken` (GPT-2 BPE tokenizer)
- `notebook` or `jupyterlab` (for running the notebook)

You can install these into a virtual environment. On Windows PowerShell:

```powershell
# Create and activate a virtual environment (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install torch tiktoken notebook
# (Optionally) pip install jupyterlab tqdm matplotlib
```

Note: For GPU acceleration, install a CUDA-enabled PyTorch build appropriate for your system (see the official PyTorch install selector). CPU-only is fine for small experiments.

## Quickstart

### 1) Run the notebook

```powershell
# From the repo root
jupyter notebook pretraining.ipynb
# or
jupyter lab pretraining.ipynb
```

The notebook walks through preparing data, training, checkpointing, and generating text from the trained model.

### 2) Use the model in a script

Below is a minimal example to instantiate the model and perform greedy generation using the included helper.

```python
import torch
import tiktoken

from model.SydsGPT import SydsGPT
from modules.GenerateSimple import generate_simple

# Minimal config (tune as needed)
config = {
    "vocab_size": 50257,        # GPT-2 tokenizer size
    "embedding_dim": 256,
    "context_length": 128,
    "num_layers": 4,
    "num_heads": 4,
    "dropout": 0.1,
    "qkv_bias": False,
}

# Tokenizer
enc = tiktoken.get_encoding("gpt2")

# Prompt
prompt = "Once upon a time"
input_ids = torch.tensor([enc.encode(prompt)], dtype=torch.long)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SydsGPT(config).to(device).eval()

# Greedy generation (argmax)
max_new_tokens = 100
context_len = config["context_length"]
with torch.no_grad():
    generated = generate_simple(model, input_ids.to(device), max_new_tokens, context_len)

print(enc.decode(generated[0].tolist()))
```

## Data

- The notebook demonstrates creating a training dataset from raw text using the GPT-2 tokenizer.
- The helper `modules/DataLoader.py` provides:
  - `create_dataloader(text, max_length=512, step_size=256, batch_size=8, ...)`
  - A tiny `Dataset` that yields `(input_ids, target_ids)` pairs shifted by one token.

## Training notes

- Loss: next-token prediction via cross-entropy on the model logits.
- Optimizer: AdamW is a typical choice (see the notebook for a full training loop).
- Checkpointing: Save and load `state_dict()` for reproducible runs; include the optimizer state if you plan to resume training.

Example checkpoint pattern:

```python
# Save
ckpt = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "config": config,
}
torch.save(ckpt, "checkpoints/sydsgpt.pt")

# Load
ckpt = torch.load("checkpoints/sydsgpt.pt", map_location=device)
model.load_state_dict(ckpt["model"])
optimizer.load_state_dict(ckpt["optimizer"])  # if resuming
```

## Decoding and generation

Included:
- `modules/GenerateSimple.py` — a greedy (argmax) decoder that repeatedly feeds back the last token.

Extensions you can try (see the notebook):
- Temperature scaling, top-k filtering, multinomial sampling
- Early stop on EOS token
- Repetition penalties or n-gram blocking

## Tips & troubleshooting

- CUDA out of memory: reduce `batch_size`, `context_length`, or `embedding_dim`.
- Diverging loss: start with a smaller model, reduce learning rate, or increase gradient clipping.
- Tokenizer mismatch: Always use the same tokenizer (`tiktoken` GPT-2) for both training and generation.
- Sequence length: Ensure your prompt length + new tokens never exceeds `context_length`; the helpers crop to the last `context_length` tokens.

## Contributing

Issues and PRs are welcome. If you add features (e.g., top-p sampling or better data pipelines), please keep the code simple and well-commented.
