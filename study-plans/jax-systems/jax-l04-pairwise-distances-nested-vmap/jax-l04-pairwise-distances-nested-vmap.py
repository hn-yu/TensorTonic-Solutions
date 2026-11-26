import jax
import jax.numpy as jnp

def pairwise_metrics(a: jax.Array, b: jax.Array) -> dict:
    """
    Returns a dict with sq_dist and cosine, both float32 arrays of shape (N, M).
    """
    def s_fn(x, y):
        return jnp.sum((x-y)**2)
    def c_fn(x, y):
        top = jnp.dot(x,y)
        bot = jnp.maximum(jnp.linalg.norm(x)*jnp.linalg.norm(y), 1e-8)
        return top/bot
    sq_dist = jax.vmap(jax.vmap(s_fn, in_axes=(None, 0)), in_axes=(0, None))(a, b).astype(jnp.float32)
    cosine = jax.vmap(jax.vmap(c_fn, in_axes=(None, 0)), in_axes=(0, None))(a, b)
    return {"sq_dist": sq_dist, "cosine":cosine}
    