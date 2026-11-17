import jax
import jax.numpy as jnp

def ema_scan(values: jax.Array, decay: jax.Array) -> dict:
    """
    Returns a dict with final_state of shape (D,) and corrected of shape (T, D), both float32.
    """
    def ema(carry, x):
        prev_m, step = carry
        m = decay*prev_m + (1-decay)*values[step-1]
        norm_m = m/(1-decay**step)
        return (m,step+1), norm_m
    final, ys = jax.lax.scan(ema, (jnp.zeros_like(values[0]), 1), values)
    return {"final_state": final[0], "corrected":ys}
