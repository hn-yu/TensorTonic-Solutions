import jax
import jax.numpy as jnp

def ensemble_predict(params: dict, x: jax.Array) -> dict:
    """
    Returns a dict with outputs of shape (M, N, O) and mean of shape (N, O), both float32.
    """
    def mlp(params, x):
        y = jax.nn.relu(x@params["w1"]+params["b1"])
        y = y@params["w2"]+params["b2"]
        return y

    outputs = jax.vmap(mlp, in_axes=(0, None))(params, x)
    mean = jnp.mean(outputs, axis=0)
    return {"outputs": outputs, "mean": mean}