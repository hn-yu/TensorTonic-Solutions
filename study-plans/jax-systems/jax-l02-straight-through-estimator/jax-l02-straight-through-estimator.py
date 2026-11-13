import jax
import jax.numpy as jnp

def ste_fake_quantize(x: jax.Array, scale: float, qmin: int, qmax: int, cotangent: jax.Array) -> dict:
    """
    Returns a dict with float32 values and grads, both shaped like x.
    """
    @jax.custom_vjp
    def f(x):
        return scale*jnp.clip(jnp.round(x/scale), qmin, qmax)

    def f_fwd(x):
        return f(x), jnp.where((qmin<= x/scale) & (x/scale<= qmax), 1, 0)

    def f_bwd(residuals,c):
        return (residuals*c,)

    f.defvjp(f_fwd, f_bwd)
    values, grad_fn = jax.vjp(f,x)
    grads,  = grad_fn(cotangent)
    return {"values": values, "grads": grads}