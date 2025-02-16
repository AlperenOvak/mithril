import numpy as np
import torch
import torch.nn.functional as F
import mithril as ml
import json
import math
from pathlib import Path
from typing import Any
from mithril.framework.common import IOKey
from mithril.models import (
    Model,
    Linear,
    Softmax,
    Concat,
    Buffer,
    Reshape,
    Arange,
    Cosine,
    Sine
)
from mithril import Backend

# Define the configuration and input
args = {
    "n_heads": 8,
    "n_kv_heads": 4,
    "head_dim": 64,
    "dim": 512,
    "rope_theta" : 10000
}

B, L, D = 2, 10, args["dim"]
x = np.random.rand(B, L, D).astype(np.float32)

def RoPE(dim: int, theta: int) -> Model:
    assert dim % 2 == 0
    block = Model()
    input = IOKey("input")
    block += Arange(start=0, stop=dim, step=2)(output="arange")

    omega = 1.0 / (theta ** (block.arange / dim))  # type: ignore
    out = input[..., None] * omega

    out_shape = out.shape
    B, N, D = out_shape[0], out_shape[1], out_shape[2]

    block += Cosine()(out, output="cos")
    block += Sine()(out, output="sin")

    block += Concat(n=4, axis=-1)(
        input1=block.cos[..., None],  # type: ignore
        input2=-block.sin[..., None],  # type: ignore
        input3=block.sin[..., None],  # type: ignore
        input4=block.cos[..., None],  # type: ignore
    )
    rope_shape = (B, N, D, 2, 2)
    block += Reshape()(shape=rope_shape, output=IOKey("output"))
    block.set_cin("input")
    return block

def repeat_kv_heads(a: ml.Connection, repeats: int, L, B ,n_heads,n_kv_heads) -> ml.Connection:
    block = Model()
    a = a.reshape((B, n_kv_heads, 1, L, -1))
    
    # Create a dictionary with proper input keys for Concat operation
    concat_inputs = {f"input{idx+1}": a for idx in range(repeats)}

    
    # Unpack the inputs for Concat
    block += Concat(n=repeats, axis=2)(**concat_inputs, output=IOKey("repeated"))    
    
    return IOKey("repeated").reshape((B, n_heads, L, -1))

# Define the llama_attention function in Mithril
def llama_attention(
    args: dict[str, Any],
    use_mask: bool = False,
    *,
    name: str | None = None,
):
    n_heads = args["n_heads"]
    n_kv_heads = args["n_kv_heads"]
    head_dim = args["head_dim"]
    dim = args["dim"]
    #rope_traditional = args["rope_traditional"] ## ???
    rope_theta = args["rope_theta"]

    repeats = n_heads // n_kv_heads
    scale = head_dim**-0.5

    block = Model(name=name)
    x = IOKey("input2_1", shape=(None, None, dim))

    block += Linear(n_heads * head_dim, name="wq", use_bias=False)(x, output="queries")
    block += Linear(n_kv_heads * head_dim, name="wk", use_bias=False)(x, output="keys")
    block += Linear(n_kv_heads * head_dim, name="wv", use_bias=False)(x, output="values")

    queries: ml.Connection = block.queries  # type: ignore
    keys: ml.Connection = block.keys  # type: ignore
    values: ml.Connection = block.values  # type: ignore

    B, L = queries.shape[0], queries.shape[1]
    queries = queries.reshape((B, L, n_heads, -1)).transpose((0, 2, 1, 3))  # type: ignore
    keys    = keys.reshape((B, L, n_kv_heads, -1)).transpose((0, 2, 1, 3))  # type: ignore
    values  = values.reshape((B, L, n_kv_heads, -1)).transpose((0, 2, 1, 3))  # type: ignore
    
    keys   = keys.reshape((B, n_kv_heads, 1, L, -1)) * repeats  # type: ignore
    block += Concat(n=repeats, axis=2)(input1=keys, output=IOKey("keys_repeated"))
    keys = block.keys_repeated.reshape((B, n_heads, L, -1))

    values = values.reshape((B, n_kv_heads, 1, L, -1)) * repeats  # type: ignore
    block += Concat(n=repeats, axis=2)(input1=values, output=IOKey("values_repeated"))
    values = block.values_repeated.reshape((B, n_heads, L, -1))

    block += RoPE(dim=head_dim, theta=rope_theta)(input=queries, output=IOKey("queriesRope_out"))
    queries = block.queriesRope_out

    block += RoPE(dim=head_dim, theta=rope_theta)(input=keys, output=IOKey("keysRope_out"))
    keys = block.keysRope_out

    scores = (queries * scale) @ keys.transpose((0, 1, 3, 2))
    if use_mask:
        scores = scores + IOKey("mask").cast(scores.dtype())

    block += Softmax(axis=-1)(scores.cast(ml.float32), output="attention_weights")

    scores = block.attention_weights.cast(scores.dtype())  # type: ignore
    output = (scores @ values).transpose((0, 2, 1, 3)).reshape((B, L, -1))
    block += Linear(dim, name="wo", use_bias=False)(output, output=IOKey("output"))
    block += Buffer()(keys, output=IOKey("keys_out"))
    block += Buffer()(values, output=IOKey("values_out"))

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
pm = ml.compile(model, backend=numpy_backend, inference=True, jit=False)

# Run the Mithril function
mithril_output = pm.evaluate({}, data={"input2_1": x})

# Run the NumPy function
numpy_output = numpy_attention(x, args)

# Compare the results
print("Mithril Output:", mithril_output)
print("NumPy Output:", numpy_output)
print("Difference:", np.abs(mithril_output - numpy_output).sum())


















"""import mithril as ml 
from mithril.models import Add, MatrixMultiply, Softmax, IOKey, Model
from mithril import Backend

#backend = ml.MlxBackend()
#jax_backend = ml.JaxBackend()
numpy_backend = ml.NumpyBackend()


model = Model()
model += Add()(IOKey('x'), IOKey('y'), output=IOKey('z'))


pm = ml.compile(model, backend=numpy_backend, inference=True, jit=False)


input1 = numpy_backend.array([1, 2, 3])
input2 = numpy_backend.array([4, 5, 6])

outputs = pm.evaluate({}, data={"x": input1, "y": input2})"""