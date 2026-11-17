import jax
import jax.numpy as jnp

def per_head_rms_norm(x: jax.Array, num_heads: int) -> dict:
    """
    Returns a dict with heads of shape (B, num_heads, T, D // num_heads) and merged of shape (B, T, D), both float32.
    """
    b, t, d = x.shape
    d_h = d//num_heads
    # x (B, T, D) -> (B, H, T, D_H)
    x = x.reshape(b,t,num_heads,d_h)
    x = jnp.swapaxes(x,1,2)
    norm = jnp.sqrt(1/d_h*jnp.sum(x ** 2, axis=-1, keepdims=True) + 1e-6)
    x_normed = x / norm
    merged = x_normed.transpose(0, 2, 1, 3)
    merged = merged.reshape(b, t, d)
    return {"heads":x_normed, "merged":merged,}

per_head_rms_norm = jax.jit(per_head_rms_norm, static_argnames=("num_heads",))