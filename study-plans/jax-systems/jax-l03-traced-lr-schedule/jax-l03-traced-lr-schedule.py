import jax
import jax.numpy as jnp

def lr_schedule(step: jax.Array, peak_lr: float, min_lr: float, warmup_steps: int, total_steps: int) -> jax.Array:
    """
    Returns the float32 learning rate for every entry of step, with the same shape as step.
    """
    warmup_steps = float(warmup_steps)
    i = jnp.where(
        step < warmup_steps,
        peak_lr*step/warmup_steps,
        min_lr + (peak_lr-min_lr)/2*
        (1+jnp.cos(jnp.pi*jnp.minimum((step-warmup_steps)/
                                      (total_steps-warmup_steps),1)))
    )
    return i

lr_schedule = jax.jit(lr_schedule, static_argnames=("peak_lr", "min_lr", "warmup_steps", "total_steps"))
