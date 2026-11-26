from typing import Any
import jax
import jax.numpy as jnp

def clip_by_global_norm(grads: Any, max_norm: float) -> dict:
    """
    Returns a dict with the clipped gradient pytree under clipped and the float32 global norm under global_norm.
    """
    leaves = jax.tree.leaves(grads)
    squared = sum(jnp.sum(jnp.square(leaf.astype(jnp.float32))) for leaf in leaves)
    g_norm = jnp.sqrt(squared).astype(jnp.float32)
    s = jnp.where(g_norm > max_norm, max_norm / g_norm, 1.0)
    clipped =  jax.tree.map(
        lambda x: (x*s).astype(x.dtype),
        grads
    )
    return {"clipped": clipped, "global_norm":g_norm}
