from typing import Any
import jax
import jax.numpy as jnp

def momentum_sgd_step(params: Any, grads: Any, velocity: Any, learning_rate: float, momentum: float) -> tuple:
    """
    Returns a tuple (new_params, new_velocity) of pytrees with the same structure as params.
    """
    v_new = jax.tree.map(
        lambda g,v: momentum*v+g,
        grads,
        velocity
    )

    p_new = jax.tree.map(
        lambda p,v: p-learning_rate*v,
            params,
            v_new
    )
    return p_new, v_new
