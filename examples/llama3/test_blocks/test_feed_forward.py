import sys
import os
from typing import Any

# Get the absolute path of the mithril directory
MITHRIL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))  # Adjust this based on your location

# Add it to sys.path
sys.path.insert(0, MITHRIL_PATH)

import numpy as np
import mithril as ml
from mithril.models import (
    Model, 
    Linear, 
    Multiply, 
    SiLU,
    IOKey,
    Buffer,
)



# ---- Mithril Implementation ----
def feed_forward(args: dict[str, Any], *, name: str | None = None):
    block = Model(name=name)
    x = IOKey("input", shape=(2, 16, dim))
    
    # Projections matching MLX's structure
    block |= Linear(args["hidden_dim"], name="w1", use_bias=False)(x, output="w1_out")
    block |= Linear(args["hidden_dim"], name="w3", use_bias=False)(x, output="w3_out")
    
    # SiLU activation and element-wise multiplication
    block |= SiLU()(block.w1_out, output="silu_out")
    block |= Multiply()(block.silu_out, block.w3_out, output="multiplied")
    
    # Final projection
    block |= Linear(args["dim"], name="w2", use_bias=False)(block.multiplied, output=IOKey("output"))
    
    return block


# ---- Testing and Comparison ----
dim = 512
hidden_dim = 1024
B, L = 2, 16  # Batch size, Sequence length

# Generate random input
x = np.random.rand(B, L, dim).astype(np.float32)

args = {
    "n_heads": 8,
    "n_kv_heads": 4,
    "hidden_dim": 1024,
    "dim": 512,
    "rope_theta": 10000.0,
}

# Initialize Mithril FeedForward
numpy_backend = ml.NumpyBackend()
mithril_model = feed_forward(args)
pm = ml.compile(mithril_model, backend=numpy_backend, inference=True, jit=False, file_path="ffn_test.py")

# Get same parameters for both implementations
params = pm.randomize_params()

# Run Mithril function
mithril_output = pm.evaluate(params, data={"input": x})

# ---- NumPy Implementation ----
def feedforward_numpy(params, x: np.ndarray):
    """
    NumPy implementation of FeedForward using parameters from Mithril.
    """
    W1 = params["weight_0"].T  # Transpose to match Mithril
    W3 = params["weight_1"].T
    W2 = params["weight_2"].T

    silu = x @ W1
    silu = silu / (1 + np.exp(-silu))  # SiLU activation
    gate = x @ W3

    return (silu * gate) @ W2
# Run NumPy function with the same params
numpy_output = feedforward_numpy(params, x)

# Compare the results
print("Mithril Output Shape:", mithril_output["output"].shape)
print("NumPy Output Shape:", numpy_output.shape)
print("Difference (Mean Absolute Error):", np.mean(np.abs(mithril_output["output"] - numpy_output)))
