import jax
import jax.numpy as jnp

def hessian_vector_product(w: jax.Array, v: jax.Array, x: jax.Array, y: jax.Array, alpha: float) -> jax.Array:
    """
    Returns the float32 Hessian-vector product of shape (D,).
    """
    def loss_fn(w):
        l2_loss = alpha/4*jnp.sum(w**4)
        loss_1 = jnp.mean(jax.nn.softplus(x@w) - x*y[:,None]@w)
        return loss_1 + l2_loss
    _, hvp = jax.jvp(jax.grad(loss_fn), (w,), (v,))
    return jnp.array(hvp, dtype=jnp.float32)