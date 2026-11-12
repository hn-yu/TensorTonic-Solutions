import jax
import jax.numpy as jnp

def per_example_clipped_gradient(params: dict, x: jax.Array, y: jax.Array, max_norm: float) -> dict:
    """
    Returns a dict with the averaged clipped grads (keys w and b) and the float32 per-example norms.
    """
    def loss_fn(params,x,y):
        return 0.5*(x@params["w"]+params["b"]-y)**2
    g_n = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))(params, x, y)
    g = jnp.concat([g_n["w"], g_n["b"][:, None]], axis=1)
    norms = jnp.linalg.norm(g, axis=1)
    factor = jnp.maximum(norms/max_norm, 1)
    clipped_w = g_n["w"]/factor[:, None]
    clipped_b = g_n["b"]/factor
    grads = {
        "w" : jnp.mean(clipped_w, axis=0),
        "b" : jnp.mean(clipped_b)
    }
    return {"grads":grads, "norms":norms}
    
    