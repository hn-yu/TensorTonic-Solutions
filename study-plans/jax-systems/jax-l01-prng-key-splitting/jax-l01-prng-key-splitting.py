import jax
import jax.numpy as jnp

def dropout_masks(seed: int, num_layers: int, shape: list[int], keep_prob: float) -> jax.Array:
    """
    Returns a bool array of shape (num_layers, followed by the entries of shape) holding one dropout keep mask per layer.
    """
    key = jax.random.key(seed)
    mask = jnp.empty((num_layers, *shape), dtype=bool)
    for i in range(num_layers):
        key, new_key = jax.random.split(key)
        mask_i = jax.random.bernoulli(new_key, keep_prob, shape)
        mask = mask.at[i].set(mask_i)
    return jnp.asarray(mask)
