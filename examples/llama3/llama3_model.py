
import numpy as np
import sys
import os

# Get the absolute path of the mithril directory
MITHRIL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))  # Adjust this based on your location

# Add it to sys.path
sys.path.insert(0, MITHRIL_PATH)

import mithril as ml
import json
import math
from pathlib import Path
from typing import Any
from mithril.models import (
    Model,
    Linear,
    SiLU,
    Embedding,
    Add,
    Arange,
    Transpose,
    Reshape,
    ScaledDotProduct,
    LayerNorm,  
    Gelu,
    Multiply,
    Softmax,
    Concat,
    Buffer,
    Split,
    Subtract,
    IOKey
    
)


def rms_norm(dim: int, *, name: str | None = None):
    # TODO: add eps parameter
    # TODO: check original implementation they use astype and cast to float32
    block = Model(name=name)
    input = IOKey("input")
    weight = IOKey(
        "weight", shape=[dim], differentiable=True
    )  # TODO: weight must be initialized with ones.
    rrms = input / ((input**2).mean(axis=-1, keepdim=True) + 1e-5).sqrt()
    block += Multiply()(left=rrms, right=weight, output=IOKey("output"))
    block.set_cin("input")
    return block


def apply_rotary_pos_emb():
    block = Model()
    xq = IOKey("xq")  # Original shape: (B, H, L, D)
    xk = IOKey("xk")
    freqs_cis = IOKey("freqs_cis")  # Shape: (L, D//2, 2)

    # Get dimensions directly from input shapes
    B, H, L, D = xq.shape[0], xq.shape[1], xq.shape[2], xq.shape[3]
    D_half = D // 2  # This must be integer since we're splitting into complex pairs

    # Reshape queries/keys to complex form
    block |= Reshape()(xq, shape=(B, H, L, D_half, 2), output="xq_")
    block |= Reshape()(xk, shape=(B, H, L, D_half, 2), output="xk_")

    # Split frequency components (L, D//2, 2) -> [(L, D//2, 1), (L, D//2, 1)]
    block |= Split(split_size=2, axis=-1)(freqs_cis, output="freqs_split")
    
    # Prepare frequency tensors for broadcasting
    # Reshape to (1, 1, L, D_half, 1) to match query/key dimensions
    block |= Reshape()(block.freqs_split[0], shape=(1, 1, L, D_half, 1), output="freqs_cos")
    block |= Reshape()(block.freqs_split[1], shape=(1, 1, L, D_half, 1), output="freqs_sin")

    # Split complex numbers into real/imaginary parts
    block |= Split(split_size=2, axis=-1)(block.xq_, output="xq_split")
    xq_real = block.xq_split[0]  # (B, H, L, D_half, 1)
    xq_imag = block.xq_split[1]

    # Apply rotary transformations
    block |= Multiply()(block.freqs_cos, xq_real, output="cos_xq_real")
    block |= Multiply()(block.freqs_sin, xq_imag, output="sin_xq_imag")
    block |= Subtract()(block.cos_xq_real, block.sin_xq_imag, output="xq_out_real")

    block |= Multiply()(block.freqs_sin, xq_real, output="sin_xq_real")
    block |= Multiply()(block.freqs_cos, xq_imag, output="cos_xq_imag")
    block |= Add()(block.sin_xq_real, block.cos_xq_imag, output="xq_out_imag")

    # Combine real/imaginary and reshape back
    xqs = {"input1":block.xq_out_real, "input2":block.xq_out_imag}
    block |= Concat(n=2, axis=-1)(**xqs, output="xq_out_combined")
    block |= Reshape()(block.xq_out_combined, shape=(B, H, L, D), output=IOKey("xq_out"))

    # Repeat steps 4-6 for keys
    block |= Split(split_size=2, axis=-1)(block.xk_, output="xk_split")
    xk_real = block.xk_split[0]
    xk_imag = block.xk_split[1]

    block |= Multiply()(block.freqs_cos, xk_real, output="cos_xk_real")
    block |= Multiply()(block.freqs_sin, xk_imag, output="sin_xk_imag")
    block |= Subtract()(block.cos_xk_real, block.sin_xk_imag, output="xk_out_real")

    block |= Multiply()(block.freqs_sin, xk_real, output="sin_xk_real")
    block |= Multiply()(block.freqs_cos, xk_imag, output="cos_xk_imag")
    block |= Add()(block.sin_xk_real, block.cos_xk_imag, output="xk_out_imag")

    xks = {"input1":block.xk_out_real, "input2":block.xk_out_imag}
    block |= Concat(n=2, axis=-1)(**xks, output="xk_out_combined")
    block |= Reshape()(block.xk_out_combined, shape=(B, H, L, D), output=IOKey("xk_out"))

    return block

# Define the llama_attention function in Mithril
def llama_attention(
    args: dict[str, Any],
    use_mask: bool = False,
    *,
    name: str | None = None,
):
    n_heads = args["n_heads"]
    n_kv_heads = args["n_kv_heads"] #pm.randomize
    head_dim = args["head_dim"]
    dim = args["dim"]
    rope_theta = args["rope_theta"]
    #rope_traditional = args["rope_traditional"] ## ???
    freqs_cis = IOKey("freqs_cis")


    repeats = n_heads // n_kv_heads
    scale = head_dim**-0.5

    block = Model(name=name)
    x = IOKey("input", shape=(2, 16, dim))

    block |= Linear(n_heads * head_dim, name="wq", use_bias=False)(x, output="queries")
    block |= Linear(n_kv_heads * head_dim, name="wk", use_bias=False)(x, output="keys")
    block |= Linear(n_kv_heads * head_dim, name="wv", use_bias=False)(x, output="values")

    queries: ml.Connection = block.queries  # type: ignore
    keys: ml.Connection = block.keys  # type: ignore
    values: ml.Connection = block.values  # type: ignore

    B, L = queries.shape[0], queries.shape[1]
    queries = queries.reshape((B, L, n_heads, -1)).transpose((0, 2, 1, 3))  # type: ignore
    keys    = keys.reshape((B, L, n_kv_heads, -1)).transpose((0, 2, 1, 3))  # type: ignore
    values  = values.reshape((B, L, n_kv_heads, -1)).transpose((0, 2, 1, 3))  # type: ignore

    
    keys   = keys.reshape((B, n_kv_heads, 1, L, -1)) # * repeats  # type: ignore
    concat_keys= {f"input{idx+1}": keys for idx in range(repeats)}
    block |= Concat(n=repeats, axis=2)(**concat_keys, output=IOKey("keys_repeated"))
    keys = block.keys_repeated.reshape((B, n_heads, L, -1))

    values = values.reshape((B, n_kv_heads, 1, L, -1)) # * repeats  # type: ignore
    concat_values= {f"input{idx+1}": values for idx in range(repeats)}
    block |= Concat(n=repeats, axis=2)(**concat_values, output=IOKey("values_repeated"))
    values = block.values_repeated.reshape((B, n_heads, L, -1))


    block |= apply_rotary_pos_emb()(
        xq=queries, xk=keys, freqs_cis=freqs_cis, xq_out="xq_out", xk_out="xk_out"
    )
    
    queries = block.xq_out
    keys = block.xk_out    

    scores = (queries * scale) @ keys.transpose((0, 1, 3, 2))
    if use_mask:
        scores = scores + IOKey("mask").cast(scores.dtype())

    block |= Softmax(axis=-1)(scores.cast(ml.float32), output="attention_weights")

    scores = block.attention_weights.cast(scores.dtype())  # type: ignore
    output = (scores @ values).transpose((0, 2, 1, 3)).reshape((B, L, -1))
    block |= Linear(dim, name="wo", use_bias=False)(output, output=IOKey("output"))
    block |= Buffer()(keys, output=IOKey("keys_out"))
    block |= Buffer()(values, output=IOKey("values_out"))

    return block

def feed_forward(args: dict[str, Any], *, name: str | None = None):
    block = Model(name=name)
    x = IOKey("input", shape=(None, None, args["dim"]))
    
    # Projections matching MLX's structure
    block |= Linear(args["hidden_dim"], name="w1", use_bias=False)(x, output="w1_out")
    block |= Linear(args["hidden_dim"], name="w3", use_bias=False)(x, output="w3_out")
    
    # SiLU activation and element-wise multiplication
    block |= SiLU()(block.w1_out, output="silu_out")
    block |= Multiply()(block.silu_out, block.w3_out, output="multiplied")
    
    # Final projection
    block |= Linear(args["dim"], name="w2", use_bias=False)(block.multiplied, output=IOKey("output"))
    
    return block


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

def llama_model(args: dict[str, Any], *, name: str | None = None):
    block = Model(name=name)
    x = IOKey("input", shape=(2, 16))  # Token indices (B, L)
    
    # 1. Token embeddings (MLX: self.tok_embeddings)
    block |= Embedding(
        num_embeddings=args["vocab_size"],
        dim=args["dim"],
        name="tok_embeddings"
    )(input=x, output="embeddings")
    
    # 2. Create causal mask input (MLX: create_additive_causal_mask)
    mask = IOKey("mask", shape=(1, 1, 16, 16))  # (1, 1, L, L) ##  TODO: Check this shape
    
    # 3. Transformer layers (MLX: self.layers)
    current = block.embeddings
    for i in range(args["n_layers"]):
        tb = transformer_block(args, use_mask=True, name=f"layer_{i}")(
            input=current,
            freqs_cis=IOKey("freqs_cis"),
            mask=mask,
            output=f"layer_{i}_out"
        )
        block |= tb
        current = tb.output
        # Cache handling (MLX: cache.append(c))
        # block |= Buffer()(tb.keys_out, output=IOKey(f"keys_{i}"))
        # block |= Buffer()(tb.values_out, output=IOKey(f"values_{i}"))
    
    # 4. Final normalization (MLX: self.norm)
    block |= rms_norm(args["dim"], name="norm")(
        input=current, 
        output="norm_out"
    )
    
    # 5. Output projection (MLX: self.output)
    block |= Linear(
        input_dim=args["dim"],
        output_dim=args["vocab_size"], 
        name="output", 
        use_bias=False
    )(input=block.norm_out, output="logits")
    
    return block
