import sys
import os

# Get the absolute path of the mithril directory
MITHRIL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))  # Adjust this based on your location

# Add it to sys.path
sys.path.insert(0, MITHRIL_PATH)

import mithril as ml
from mithril.models import Model, Linear, Buffer, IOKey

class MultiHeadAttention(ml.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = Linear(dim, dim, name="q_proj", use_bias=False)
        self.k_proj = Linear(dim, dim, name="k_proj", use_bias=False)
        self.v_proj = Linear(dim, dim, name="v_proj", use_bias=False)
        self.out_proj = Linear(dim, dim, name="out_proj", use_bias=False)
        
    def __call__(self, 
                queries: IOKey, 
                keys: IOKey, 
                values: IOKey,
                mask: Optional[IOKey] = None):
        block = Model()
        
        # Projections
        block |= self.q_proj(queries, output="q")
        block |= self.k_proj(keys, output="k")
        block |= self.v_proj(values, output="v")
        
        # Reshape for multi-head attention
        B, L_q, D = queries.shape
        block |= Reshape()(block.q, shape=(B, L_q, self.n_heads, -1), output="q_reshape")
        block |= Transpose()(block.q_reshape, perm=(0, 2, 1, 3), output="q_transposed")
        
        _, L_k, _ = keys.shape
        block |= Reshape()(block.k, shape=(B, L_k, self.n_heads, -1), output="k_reshape")
        block |= Transpose()(block.k_reshape, perm=(0, 2, 1, 3), output="k_transposed")
        
        block |= Reshape()(block.v, shape=(B, L_k, self.n_heads, -1), output="v_reshape")
        block |= Transpose()(block.v_reshape, perm=(0, 2, 1, 3), output="v_transposed")
        
        # Scaled dot-product attention
        block |= MatMul()(block.q_transposed, block.k_transposed.transpose((0, 1, 3, 2)), output="scores")
        block |= Multiply()(block.scores, self.scale, output="scaled_scores")
        
        if mask is not None:
            block |= Add()(block.scaled_scores, mask, output="scores_masked")
        
        block |= Softmax(axis=-1)(block.scores_masked if mask else block.scaled_scores, output="attn_weights")
        block |= MatMul()(block.attn_weights, block.v_transposed, output="attn_output")
        
        # Reshape back
        block |= Transpose()(block.attn_output, perm=(0, 2, 1, 3), output="attn_transposed")
        block |= Reshape()(block.attn_transposed, shape=(B, L_q, -1), output="attn_reshaped")
        
        # Final projection
        block |= self.out_proj(block.attn_reshaped, output="output")
        
        # Buffer for potential caching
        block |= Buffer()(block.k_transposed, output="keys_cache")
        block |= Buffer()(block.v_transposed, output="values_cache")
        
        return block