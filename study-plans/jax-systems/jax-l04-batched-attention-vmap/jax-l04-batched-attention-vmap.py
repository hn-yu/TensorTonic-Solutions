import jax
import jax.numpy as jnp

def batched_attention(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    """
    Returns the attention output of shape (B, H, Tq, Dv) with the dtype of q.
    """
    def attention(q,k,v):
        d = q.shape[-1]
        att = jax.nn.softmax(q@k.T/jnp.sqrt(d).astype(q.dtype), axis=-1)@v
        return att
    att_fn = jax.vmap(jax.vmap(attention))
    att = att_fn(q,k,v)
    return att
