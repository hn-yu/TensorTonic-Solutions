import jax
import jax.numpy as jnp

def logistic_value_and_grad(params: dict, x: jax.Array, y: jax.Array, l2: float) -> dict:
    """
    Returns a dict with the float32 scalar loss and a grads dict with keys w and b.
    """
    def loss_fn(params, x, y, l2):
        logit = x@params["w"] + params["b"]
        y_pred = jax.nn.softplus(logit)
        loss1 = jnp.mean(y_pred - y*logit)
        loss2 = 0.5*l2*jnp.sum(params["w"]**2)
        loss = loss1 + loss2
        return loss
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y, l2)
    output =  {"loss":loss, "grads":grads}
    return output
    
    