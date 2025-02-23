import numpy as np
import torch
import torch.nn.functional as F

import sys
import os

# Get the absolute path of the mithril directory
MITHRIL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))  # Adjust this based on your location

# Add it to sys.path
sys.path.insert(0, MITHRIL_PATH)

from examples.llama3.test_blocks.test_feed_forward import rms_norm, feed_forward
from examples.llama3.test_blocks.test_attention import llama_attention
import mithril as ml
import json
import math
from pathlib import Path
from typing import Any
from mithril import IOKey
from mithril.models import (
    Model,
    Linear,
    Softmax,
    Concat,
    Buffer,
    Reshape,
    Arange,
    Cosine,
    Sine,
    Multiply,
    Add,
    Split,
    Transpose,
    Subtract,
)
from mithril import Backend

def transformer_block(args: dict[str, Any], use_mask: bool = False, *, name: str | None = None):
    block = Model(name=name)
    x = IOKey("input", shape=(2, 16, args["dim"]))  # Match your attention input shape
    
    # 1. Attention normalization
    block |= rms_norm(args["dim"], name="attention_norm")(input=x, output="norm1")
    
    # 2. Apply attention with potential mask
    llama_attn = llama_attention(args, use_mask=use_mask)(
        input = block.norm1, 
        freqs_cis=IOKey("freqs_cis"),  # Connect freqs_cis from external input
        output="attn_out"
    )
    block |= llama_attn
    
    # 3. First residual connection
    block |= Add()(x, block.attn_out, output="h_res")
    
    # 4. FFN normalization
    block |= rms_norm(args["dim"], name="ffn_norm")(input=block.h_res, output="norm2")
    
    # 5. Apply feed forward
    block |= feed_forward(args)(input=block.norm2, output="ffn_out")
    
    # 6. Second residual connection
    block |= Add()(block.h_res, block.ffn_out, output="output")
    #attn_out = block.attn_out
    
    # 7. Buffer layers for potential cache (matches MLX's return pattern)
    block |= Buffer()(llama_attn.model.keys_out, output=IOKey("keys_out")) # Why do we need to use .model here? 
    block |= Buffer()(llama_attn.model.values_out, output=IOKey("values_out"))

    return block

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


# Define arguments
args = {
    "dim": 512,
    "hidden_dim": 2048,
    "n_heads": 8,
    "n_kv_heads": 4,
    "head_dim": 64,
    "norm_eps": 1e-5,
    "rope_theta": 10000.0
}

# Initialize block
block = transformer_block(args)
# Compile with buffers for cache
pm = ml.compile(
        block, 
        backend=ml.NumpyBackend(),
        inference=True,
        jit=False,
        file_path="transformer_block.py"
        )

params = pm.randomize_params()
data = {
    "input": np.random.randn(2, 16, 512).astype(np.float32),
    "freqs_cis": precompute_freqs_cis(args["head_dim"], 16)
}
output = pm.evaluate(params, data)