import numpy as np
import torch
import torch.nn.functional as F

import sys
import os

# Get the absolute path of the mithril directory
MITHRIL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))  # Adjust this based on your location

# Add it to sys.path
sys.path.insert(0, MITHRIL_PATH)

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

# Example usage
head_dim = 64  # Head dimension
seq_len = 16  # Maximum sequence length
freqs_cis = precompute_freqs_cis(head_dim, seq_len)

# Define the configuration and input
args = {
    "n_heads": 8,
    "n_kv_heads": 4,
    "head_dim": 64,
    "dim": 512,
    "rope_theta": 10000.0,
}

B, L, D = 2, 16, args["dim"]
x = np.random.rand(B, L, D).astype(np.float32)

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

    block |= Multiply()(block.freqs_cos, xk_real, output="sin_xk_real")
    block |= Multiply()(block.freqs_sin, xk_imag, output="cos_xk_imag")
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
    
    keys   = keys.reshape((B, n_kv_heads, 1, L, -1)) * repeats  # type: ignore
    concat_keys= {f"input{idx+1}": keys for idx in range(repeats)}
    block |= Concat(n=repeats, axis=2)(**concat_keys, output=IOKey("keys_repeated"))
    keys = block.keys_repeated.reshape((B, n_heads, L, -1))

    values = values.reshape((B, n_kv_heads, 1, L, -1)) * repeats  # type: ignore
    concat_values= {f"input{idx+1}": values for idx in range(repeats)}
    block |= Concat(n=repeats, axis=2)(**concat_values, output=IOKey("values_repeated"))
    values = block.values_repeated.reshape((B, n_heads, L, -1))

    block |= apply_rotary_pos_emb()(
        xq=queries, xk=keys, freqs_cis=freqs_cis, xq_out="xq_out", xk_out="xk_out"
    )

    """block |= rope(dim, rope_theta, name="rope")(block.xq_out, output="queries")
    block |= rope(dim, rope_theta, name="rope")(block.xk_out, output="keys")"""
    
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

# Define the same logic in NumPy
def numpy_attention(x, args):
    n_heads = args["n_heads"]
    n_kv_heads = args["n_kv_heads"]
    head_dim = args["head_dim"]
    dim = args["dim"]

    repeats = n_heads // n_kv_heads
    scale = head_dim**-0.5

    wq = np.random.rand(dim, n_heads * head_dim).astype(np.float32)
    wk = np.random.rand(dim, n_kv_heads * head_dim).astype(np.float32)
    wv = np.random.rand(dim, n_kv_heads * head_dim).astype(np.float32)
    wo = np.random.rand(n_heads * head_dim, dim).astype(np.float32)

    queries = np.dot(x, wq)
    keys = np.dot(x, wk)
    values = np.dot(x, wv)

    B, L, _ = queries.shape
    queries = queries.reshape(B, L, n_heads, -1).transpose(0, 2, 1, 3)
    keys = keys.reshape(B, L, n_kv_heads, -1).transpose(0, 2, 1, 3)
    values = values.reshape(B, L, n_kv_heads, -1).transpose(0, 2, 1, 3)

    def repeat(a, repeats):
        expanded = np.concatenate([np.expand_dims(a, 2)] * repeats, axis=2)
        return expanded.reshape(B, n_heads, L, -1)

    keys = repeat(keys, repeats)
    values = repeat(values, repeats)

    scores = np.matmul(queries * scale, keys.transpose(0, 1, 3, 2))
    scores = F.softmax(torch.tensor(scores), dim=-1).numpy()

    output = np.matmul(scores, values).transpose(0, 2, 1, 3).reshape(B, L, -1)
    output = np.dot(output, wo)

    return output

numpy_backend = ml.NumpyBackend()
model = llama_attention(args)
pm = ml.compile(model, backend=numpy_backend, inference=True, jit=False, file_path="test.py")
params = pm.randomize_params()
# Run the Mithril function
mithril_output = pm.evaluate(params, data={"input": x, "freqs_cis": freqs_cis})

# Define the NumPy function
def attention_numpy(params, args, x: np.ndarray, freqs_cis: np.ndarray):
    B, L, D = x.shape
    n_heads = args["n_heads"]
    n_kv_heads = args["n_kv_heads"]
    head_dim = args["head_dim"]
    repeats = n_heads // n_kv_heads
    scale = head_dim ** -0.5

    # Extract weights from Mithril params
    Wq = params["weight_0"].T  # Transpose to match Mithril's matrix multiplication
    Wk = params["weight_1"].T
    Wv = params["weight_2"].T
    Wo = params["weight_3"].T

    # Projections using Mithril's weights
    queries = x @ Wq  # (B, L, n_heads * head_dim)
    keys = x @ Wk     # (B, L, n_kv_heads * head_dim)
    values = x @ Wv   # (B, L, n_kv_heads * head_dim)

    # Reshape and transpose (matches Mithril's reshape/transpose)
    queries = queries.reshape(B, L, n_heads, head_dim).transpose(0, 2, 1, 3)
    keys = keys.reshape(B, L, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
    values = values.reshape(B, L, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

    # Repeat KV heads (matches Mithril's concat approach)
    keys = np.repeat(keys, repeats, axis=1)
    values = np.repeat(values, repeats, axis=1)

    # Apply RoPE (must match Mithril's implementation)
    def apply_rope_numpy(x, freqs_cis):
        x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
        x_real = x_reshaped[..., 0]
        x_imag = x_reshaped[..., 1]
        
        cos_theta = freqs_cis[..., 0]
        sin_theta = freqs_cis[..., 1]
        
        # Add broadcast dimensions to match Mithril's reshape
        cos_theta = cos_theta.reshape(1, 1, *cos_theta.shape)
        sin_theta = sin_theta.reshape(1, 1, *sin_theta.shape)
        
        x_real_rot = cos_theta * x_real - sin_theta * x_imag
        x_imag_rot = sin_theta * x_real + cos_theta * x_imag
        return np.stack([x_real_rot, x_imag_rot], axis=-1).reshape(x.shape)

    queries = apply_rope_numpy(queries, freqs_cis)
    keys = apply_rope_numpy(keys, freqs_cis)

    # Attention calculation
    scores = (queries * scale) @ keys.transpose(0, 1, 3, 2)
    scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    scores = scores / np.sum(scores, axis=-1, keepdims=True)

    output = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
    return output @ Wo  # Final projection with Mithril's weights

# Run the NumPy function
numpy_output = attention_numpy(params, args,x , freqs_cis)

# Compare the results
print("Mithril Output:", mithril_output)
print("NumPy Output:", numpy_output)
print("Difference:", np.mean(np.abs(mithril_output["output"] - numpy_output)))


















"""import mithril as ml 
from mithril.models import Add, MatrixMultiply, Softmax, IOKey, Model
from mithril import Backend

#backend = ml.MlxBackend()
#jax_backend = ml.JaxBackend()
numpy_backend = ml.NumpyBackend()


model = Model()
model |= Add()(IOKey('x'), IOKey('y'), output=IOKey('z'))


pm = ml.compile(model, backend=numpy_backend, inference=True, jit=False)


input1 = numpy_backend.array([1, 2, 3])
input2 = numpy_backend.array([4, 5, 6])

outputs = pm.evaluate({}, data={"x": input1, "y": input2})"""