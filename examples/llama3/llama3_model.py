
from numpy import expand_dims
import mithril as ml
import json
import math
from pathlib import Path
from typing import Any
from mithril.framework.common import IOKey
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
    
)
from mithril.utils import tree_unflatten


def RMSNorm(dim: int, name: str | None = None):
    # TODO: check original implementation they use astype and cast to float32
    input = IOKey("input")
    scale = IOKey("scale", shape=[dim])  # TODO: scale must be initialized with ones.
    rrms = 1 / ((input**2).mean(axis=-1, keepdim=True) + 1e-6).sqrt()
    # NOTE: Temporarily, we have to use Buffer to attach the functional connections
    # to the model. This is a workaround for the current limitation of the API.
    block = Model(name=name)
    block += Buffer()(rrms, output=IOKey("rrms"))
    block += Buffer()(input * rrms * scale, output=IOKey("output"))

    return block


def RoPE() -> Model:
    block = Model()
    # We define the input connections
    xq = IOKey("xq")
    xk = IOKey("xk")
    freqs_cis = IOKey("freqs_cis")

    xq_shape = xq.shape
    xk_shape = xk.shape
    B, L, H = xq_shape[0], xq_shape[1], xq_shape[2]
    block += Reshape()(xq, shape=(B, L, H, -1, 1, 2), output="xq_")
    B, L, H = xk_shape[0], xk_shape[1], xk_shape[2]
    # B,L,H = *xk.shape is not supported yet.
    block += Reshape()(xk, shape=(B, L, H, -1, 1, 2), output="xk_")
    # Do the math
    xq_out = (
        freqs_cis[..., 0] * block.xq_[..., 0] + freqs_cis[..., 1] * block.xq_[..., 1]  # type: ignore[attr-defined]
    )
    xk_out = (
        freqs_cis[..., 0] * block.xk_[..., 0] + freqs_cis[..., 1] * block.xk_[..., 1]  # type: ignore[attr-defined]
    )

    # We are explicitly defining the output connections with IOKey
    block += Reshape()(xq_out, shape=xq_shape, output=IOKey("xq_out"))
    block += Reshape()(xk_out, shape=xk_shape, output=IOKey("xk_out"))
    return block

def RepeatBlock(repeats: int, axis: int, new_shape: list, name: str = None) -> Model:
    """
    Creates a Mithril Model block that simulates a repeat operation along a specified axis.

    Args:
        repeats (int): Number of times to repeat the values along the axis.
        axis (int): The axis of the input tensor along which to repeat.
                    For an input shape of [B, L, D] and repeating along L, use axis=1.
        new_shape (list): The final shape of the tensor after repeating.
                          For example, if input is [B, L, D] and you repeat axis 1 by 3 times,
                          then new_shape should be [B, L * 3, D].
        name (str, optional): The name of the block.
    
    Returns:
        Model: A Mithril model block that outputs the repeated tensor.
    """
    block = Model(name=name)
    
    # Step 1: Expand the dimensions.
    # Since there is no dedicated ExpandDims, we simulate it using Reshape.
    # For an input of shape [B, L, D] and repeating along axis=1,
    # we insert a singleton dimension after axis 1 so that the expanded shape is:
    #    [B, L, 1, D]
    #
    # Here we assume the input shape is known. In this example, we use -1 for symbolic dimensions.
    expanded_shape = [-1, -1, 1, -1]
    block += Reshape(name="expand")(
        input=IOKey("input"),
        output=IOKey("expanded"),
        shape=expanded_shape
    )
    
    # Step 2: Duplicate the expanded tensor.
    # We use the Buffer operator as a simple identity to “copy” the tensor.
    # Create one copy per repetition.
    copy_keys = []
    for i in range(repeats):
        copy_name = f"copy_{i}"
        block += Buffer(name=copy_name)(
            IOKey("expanded"),
            output=IOKey(copy_name)
        )
        copy_keys.append(IOKey(copy_name))
    
    # Step 3: Concatenate the copies along the newly created axis.
    # The expanded tensor has shape [B, L, 1, D]. Repeating along axis=1 means we need to
    # concatenate along the dimension we inserted (i.e. axis = axis + 1; here, axis 2).
    block += Concat(axis=axis + 1, name="concat")(
        inputs=copy_keys,
        output=IOKey("tiled")
    )
    
    # Step 4: Reshape the concatenated tensor to merge the repeated axis back.
    # After concatenation, for an input originally of shape [B, L, D] and repeats=3,
    # the tensor now has shape [B, L, 3, D]. We want to reshape it to [B, L * 3, D].
    block += Reshape(name="reshape")(
        input=IOKey("tiled"),
        output=IOKey("output"),
        shape=new_shape
    )
    
    return block

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
    rope_traditional = args["rope_traditional"]
    rope_theta = args["rope_theta"]

    repeats = n_heads // n_kv_heads
    scale = head_dim**-0.5

    block = Model(name=name)
    x = IOKey("x", shape=(None, None, dim))

    block += Linear(n_heads * head_dim, name="wq", use_bias=False)(x, output="queries")
    block += Linear(n_kv_heads * head_dim, name="wk", use_bias=False)(x, output="keys")
    block += Linear(n_kv_heads * head_dim, name="wv", use_bias=False)(x, output="values")

    queries: ml.Connection = block.queries  # type: ignore
    keys: ml.Connection = block.keys  # type: ignore
    values: ml.Connection = block.values  # type: ignore

    B, L = queries.shape[0], queries.shape[1]
    queries = queries.reshape((B, L, n_heads, -1)).transpose((0, 2, 1, 3))  # type: ignore
    keys = keys.reshape((B, L, n_kv_heads, -1)).transpose((0, 2, 1, 3))  # type: ignore
    values = values.reshape((B, L, n_kv_heads, -1)).transpose((0, 2, 1, 3))  # type: ignore
    
    def repeat(a, repeats):
        expanded = [a.expand_dims(2)] * repeats
        block += Concat(n=repeats, axis=2)(*expanded, output="repeated")
        return block.repeated.reshape((B, n_heads, L, -1))
    
    keys, values = map(repeat, (keys, values))

    block += RoPE()(
        xq=queries, xk=keys, freqs_cis="pe", xq_out="queries_out", xk_out="keys_out"
    )
    queries = block.queries_out
    keys = block.keys_out

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

def llama_feed_forward(name: str, dim: int, hidden_dim: int):
    """
    Implements the FFN: output = Linear(hidden_dim, dim)( SiLU(Linear(dim, hidden_dim)(x)) * Linear(dim, hidden_dim)(x) )
    """
    ffn = Model(name=name)
    ffn += Linear(dim, hidden_dim, name="w1", use_bias=False)(input="input", output="ffn_w1")
    ffn += SiLU(name="silu")(input="ffn_w1", output="ffn_silu")
    ffn += Linear(dim, hidden_dim, name="w3", use_bias=False)(input="input", output="ffn_w3")
    ffn += Multiply(name="mul")(["ffn_silu", "ffn_w3"], output="ffn_mul")
    ffn += Linear(hidden_dim, dim, name="w2", use_bias=False)(input="ffn_mul", output="ffn_out")
    return ffn


def transformer_block(
    name: str,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    hidden_dim: int,
    norm_eps: float,
    rope_theta: float,
    rope_traditional: bool = True,
):
    """
    A single transformer block: (attention + residual) then (FFN + residual).
    Uses RMSNorm before attention and before the FFN.
    """
    block = Model(name=name)
    # Attention sub-block
    block += RMSNorm(dim, eps=norm_eps, name="attn_norm")(input="input", output="norm1")
    block += llama_attention(
        "attention", dim, n_heads, n_kv_heads, head_dim, rope_theta, rope_traditional
    )(input="norm1", output="attn_out")
    block += Add(name="residual1")(["input", "attn_out"], output="res1")
    block += RMSNorm(dim, eps=norm_eps, name="ffn_norm")(input="res1", output="norm2")
    block += llama_feed_forward("ffn", dim, hidden_dim)(input="norm2", output="ffn_out")
    block += Add(name="residual2")(["res1", "ffn_out"], output="output")
    return block


def llama_model(
    vocab_size: int,
    dim: int,
    n_layers: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    hidden_dim: int,
    norm_eps: float,
    rope_theta: float,
    rope_traditional: bool = True,
):
    model = Model(name="llama")
    
    # Token and positional embeddings
    model += Embedding(name="tok_embeddings", num_embeddings=vocab_size, dim=dim)(
        input="input_ids", output="embedded_tokens"
    )
    
    # Transformer blocks
    layers = Model(name="layers")
    for i in range(n_layers):
        layers += transformer_block(
            name=f"layer_{i}",
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            hidden_dim=hidden_dim,
            norm_eps=norm_eps,
            rope_theta=rope_theta,
            rope_traditional=rope_traditional,
        )(input="layer_in", output="layer_out")
        layers.set_cin("layer_in")
    
    model += layers(input="embedded_tokens")
    
    # Final normalization and output layer
    model += RMSNorm(dim, eps=norm_eps, name="final_norm")(input="layer_out", output="norm_out")
    model += Linear(dim, vocab_size, name="output_layer")(input="norm_out", output="logits")
    
    return model
