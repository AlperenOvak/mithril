# Copyright 2022 Synnada, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import json
import math
import numpy as np
import time
from pathlib import Path
from typing import Any
import torch
from safetensors import safe_open

from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer
from sentencepiece import SentencePieceProcessor
import sys
import os

import sys
import os

# Get the absolute path of the mithril directory
MITHRIL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))  # Adjust this based on your location

# Add it to sys.path
sys.path.insert(0, MITHRIL_PATH)

from llama3_model import llama_model
import mithril as ml
from mithril.models import (
    PhysicalModel,
)


# Get the absolute path of the mithril directory
MITHRIL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))  # Adjust this based on your location

# Add it to sys.path
sys.path.insert(0, MITHRIL_PATH)

import mithril as ml
from mithril import IOKey

class Tokenizer:
    def __init__(self, backend, model_name: str = "meta-llama/Llama-3.2-1b-Instruct"):
        """Loads the tokenizer from Hugging Face."""
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backend = backend

    def encode(self, s: str):
        return self.backend.array(
            self._tokenizer(
                s,
                return_tensors="np",
                return_attention_mask=False,
            )["input_ids"]
        )

    def decode(self, token_ids, with_sep=True):
        """Decodes token IDs back into text."""
        text = self._tokenizer.decode(token_ids, skip_special_tokens=True)
        return text if with_sep else text.strip()

    def bos_id(self):
        """Returns Beginning of Sequence token ID."""
        return self._tokenizer.bos_token_id

    def eos_id(self):
        """Returns End of Sequence token ID."""
        return self._tokenizer.eos_token_id

    def pad_id(self):
        """Returns Padding token ID."""
        return self._tokenizer.pad_token_id
    
def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    """
    Compute rotary position embeddings as complex exponentials for LLaMA-style RoPE using NumPy.

    Args:
        dim (int): Dimension of the model head.
        seq_len (int): Maximum sequence length.
        theta (float): Base frequency scaling factor.

    Returns:
        np.ndarray: Precomputed frequencies of shape [seq_len, dim // 2, 2].
    """
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2) / dim))  # Shape: [dim // 2]
    t = np.arange(seq_len)[:, None]  # Shape: [seq_len, 1]
    freqs_theta = t * freqs  # Shape: [seq_len, dim // 2]

    # Convert to cosine and sine components
    freqs_cis = np.stack([np.cos(freqs_theta), np.sin(freqs_theta)], axis=-1)  # Shape: [seq_len, dim // 2, 2]

    return freqs_cis.astype(np.float32)

def sanitize_config(config, weights):
    config.pop("model_type", None)
    n_heads = config["n_heads"]
    if "n_kv_heads" not in config:
        config["n_kv_heads"] = n_heads
    if "head_dim" not in config:
        config["head_dim"] = config["dim"] // n_heads
    if "hidden_dim" not in config:
        config["hidden_dim"] = weights["layers.0.feed_forward.w1.weight"].shape[0]
    if config.get("vocab_size", -1) < 0:
        config["vocab_size"] = weights["output.weight"].shape[-1]
    if "rope_theta" not in config:
        config["rope_theta"] = 10000
    unused = ["multiple_of", "ffn_dim_multiplier"]
    for k in unused:
        config.pop(k, None)
    return config

def load_weights(model_path: str):
    """Loads LLaMA weights and config from the specified model directory."""
    model_path = Path(model_path)

    # Load config
    with open(model_path / "config.json", "r") as f:
        config = json.load(f)

    # Load weights
    weights = {}
    weight_files = list(model_path.glob("weights.pth"))
    
    if not weight_files:
        raise FileNotFoundError(f"No weight files found in {model_path}")

    print(f"[INFO] Loading weights from {model_path}")

    for wf in weight_files:
        shard_weights = torch.load(wf, map_location="cpu")
        weights.update(shard_weights)

    config = sanitize_config(config, weights)    

    # Rename weights to Mithril format
    for key in list(weights.keys()):
        weights[key.replace(".", "_")] = weights.pop(key)  # Convert '.' to '_'

    return weights, config

"""def generate(
    prompt: str,
    model: PhysicalModel,
    tokenizer: Tokenizer,
    weights: dict,
    backend: ml.Backend,
    temp: float = 0.0
):
    def sample(logits):
        if temp == 0:
            return logits.argmax(axis=-1)
        else:
            return backend.random.categorical(logits * (1/temp))

    x = backend.array([[tokenizer.bos_id] + tokenizer.encode(prompt)])
    cache = None
    
    # Process prompt
    mask = backend.triu(backend.full((x.shape[1], x.shape[1]), -float("inf")), diagonal=1)
    output = model.evaluate(weights, {"input": x, "mask": mask})["output"]
    y = sample(output[:, -1, :])
    
    yield y
    
    # Generate tokens
    while True:
        output = model.evaluate(weights, {"input": y[:, None]})["output"]
        y = sample(output[:, -1, :])
        yield y"""

def run(prompt: str, backend: ml.Backend, model_name: str = "/home/vboxuser/Documents/mithril/examples/llama3/mlx/load_weights"):

    
    weights, config = load_weights(model_name)
    
    #tokenizer = Tokenizer(backend,  model_name )
    print("[INFO] Loaded tokenizer")
    print(config)

    # Compile model
    llama_lm = llama_model(config)
    llama_pm = ml.compile(
        llama_lm,
        backend,
        data_keys={"input"},
        shapes={"input": [1, None]},
        jit=True,
        use_short_namings=True
    )
    
    print(f"Prompt: {prompt}")
    print("Generated:", end=" ", flush=True)
    
    for token in generate(prompt, llama_pm, tokenizer, weights, backend):
        if token.item() == tokenizer.eos_id:
            break
        print(tokenizer.decode([token.item()]), end="", flush=True)
    print()

# Example usage
if __name__ == "__main__":
    run("The meaning of life is", ml.TorchBackend(dtype=ml.bfloat16))


# NOTE:
# no need to use shards, it is small enough