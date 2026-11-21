import jax
import jax.numpy as jnp

def spectral_norm(w: jax.Array, v0: jax.Array, num_iters: jax.Array) -> dict:
    """
    Returns a dict with sigma of shape (), u of shape (M,) and v of shape (N,), all float32.
    """
    def body(i,v):
        u = w@v/jnp.linalg.norm(w@v)
        new_v = w.T@u/jnp.linalg.norm(w.T@u)
        return new_v

    v = jax.lax.fori_loop(0, num_iters, body, v0)
    v = v/jnp.linalg.norm(v)
    sigma = jnp.linalg.norm(w@v)
    u = w@v/sigma
    return {"u":u, "v":v, "sigma":sigma}
    