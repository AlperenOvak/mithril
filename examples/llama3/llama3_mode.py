
from numpy import expand_dims
from mithril.framework.common import IOKey
from mithril.models import (
    Model,
    Linear,
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
    name: str,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    rope_theta: float,
    rope_traditional: bool = True,
):
    """
    Builds the attention block in Mithril.
    This function projects the input to Q, K, V, reshapes them, applies RoPE,
    computes scaled dot-product attention and then projects the result back.
    """
    attn = Model(name=name)
    # --- Projections for Q, K, V ---
    attn += Linear(dim, n_heads * head_dim, name="wq", use_bias=False)(input="input", output="q")
    attn += Linear(dim, n_kv_heads * head_dim, name="wk", use_bias=False)(input="input", output="k")
    attn += Linear(dim, n_kv_heads * head_dim, name="wv", use_bias=False)(input="input", output="v")

    # --- Reshape and transpose projections ---
    # For queries: [B, L, n_heads, head_dim] -> transpose to [B, n_heads, L, head_dim]
    attn += Reshape((-1, -1, n_heads, head_dim), name="reshape_q")(input="q", output="q_reshaped")
    attn += Transpose((0, 2, 1, 3), name="transpose_q")(input="q_reshaped", output="q_transposed")

    # For keys and values: [B, L, n_kv_heads, head_dim] -> transpose to [B, n_kv_heads, L, head_dim]
    attn += Reshape((-1, -1, n_kv_heads, head_dim), name="reshape_k")(input="k", output="k_reshaped")
    attn += Transpose((0, 2, 1, 3), name="transpose_k")(input="k_reshaped", output="k_transposed")
    attn += Reshape((-1, -1, n_kv_heads, head_dim), name="reshape_v")(input="v", output="v_reshaped")
    attn += Transpose((0, 2, 1, 3), name="transpose_v")(input="v_reshaped", output="v_transposed")

    # --- Repeat keys and values to match number of heads ---
    repeats = n_heads // n_kv_heads
    attn += RepeatBlock(repeats=repeats, axis=1, name="repeat_k")(input="k_transposed", output="k_repeated")
    attn += RepeatBlock(repeats=repeats, axis=1, name="repeat_v")(input="v_transposed", output="v_repeated")

    # --- Apply RoPE to queries and keys ---
    attn += RoPE(head_dim, traditional=rope_traditional, base=rope_theta, name="rope_q")(
        input="q_transposed", output="q_rope"
    )
    attn += RoPE(head_dim, traditional=rope_traditional, base=rope_theta, name="rope_k")(
        input="k_repeated", output="k_rope"
    )

    # --- Scale queries ---
    scale = head_dim ** -0.5
    attn += Multiply(scalar=scale, name="scale_q")(input="q_rope", output="q_scaled")

    # --- Compute attention scores ---
    # Transpose keys for dot-product: [B, n_heads, L, head_dim] -> [B, n_heads, head_dim, L]
    attn += Transpose((0, 1, 3, 2), name="transpose_k_for_attn")(
        input="k_rope", output="k_for_attn"
    )
    # Compute dot-product attention with causal masking.
    attn += ScaledDotProduct(is_causal=True, name="scaled_dot")(
        query="q_scaled", key="k_for_attn", value="v_repeated", output="attn_output"
    )

    # --- Reshape back to [B, L, n_heads * head_dim] ---
    attn += Transpose((0, 2, 1, 3), name="transpose_back")(
        input="attn_output", output="attn_transposed"
    )
    attn += Reshape((-1, -1, n_heads * head_dim), name="reshape_out")(
        input="attn_transposed", output="attn_reshaped"
    )

    # --- Final linear projection ---
    attn += Linear(n_heads * head_dim, dim, name="wo", use_bias=False)(
        input="attn_reshaped", output="output"
    )
    return attn