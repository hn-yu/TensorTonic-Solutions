import jax
import jax.numpy as jnp

def masked_mean(x: jax.Array, mask: jax.Array, axis: int) -> jax.Array:
    """
    Returns a float32 array with the shape of x without the given axis, holding the mean of the valid entries.
    """
    x_masked = jnp.where(mask, x, 0.0)
    count = jnp.sum(mask, axis)
    x_masked_sum = jnp.sum(x_masked, axis)
    mask_sum = jnp.maximum(count, 1)
    mean = x_masked_sum / mask_sum
    return mean 
