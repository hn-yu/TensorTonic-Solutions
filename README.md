# TensorTonic Solutions

Welcome to my TensorTonic solutions repository!

Here you'll find my solutions to various machine learning and deep learning problems from [TensorTonic](https://tensortonic.com).

## What is TensorTonic?

TensorTonic is a platform where you can implement core algorithms of Machine Learning from scratch.

This repository contains my personal solutions to these problems, automatically synchronized from the platform.

<!-- tensortonic:start -->
# Haonan Yu's TensorTonic Solutions

Verified machine learning implementations completed on [TensorTonic](https://www.tensortonic.com).

<p align="center">
  <img src="https://www.tensortonic.com/api/badge/yhn90116.svg" alt="TensorTonic Verified Solutions" width="100%" />
</p>

| Problem | Description | Link |
|---|---|---|
| Functional Array Update | Build a weighted histogram in JAX with indexed updates that return new arrays, so repeated bins accumulate and out-of-range indices are ignored under jit. | https://www.tensortonic.com/problems/jax-l01-functional-array-update |
| Masked Mean Without Boolean Indexing | Compute a masked mean in JAX with jnp.where so every shape stays static under jit, returning 0 for slices with no valid entries. | https://www.tensortonic.com/problems/jax-l01-masked-mean |
| PRNG Key Splitting | Generate reproducible per-layer dropout masks in JAX from one seed by splitting PRNG keys explicitly and never reusing a key. | https://www.tensortonic.com/problems/jax-l01-prng-key-splitting |
| Hessian-Vector Product | Compute exact Hessian-vector products in JAX by composing jax.jvp with jax.grad, without ever building the full Hessian. | https://www.tensortonic.com/problems/jax-l02-hessian-vector-product |
| Per-Example Gradient Clipping | Get per-example gradients with jax.vmap over jax.grad, clip each to a maximum norm and average them, the core of a DP-SGD step. | https://www.tensortonic.com/problems/jax-l02-per-example-gradient-clipping |
| Straight-Through Estimator with Custom VJP | Write a rounding quantizer with jax.custom_vjp whose backward pass is a clipped straight-through estimator, as used in quantization-aware training. | https://www.tensortonic.com/problems/jax-l02-straight-through-estimator |
| Value and Grad of a Loss | Use jax.value_and_grad to get a logistic regression loss and the gradients of a parameter dict in one forward and backward pass. | https://www.tensortonic.com/problems/jax-l02-value-and-grad-loss |
| Exponential Moving Average with Scan | Compute a bias-corrected exponential moving average with jax.lax.scan, returning the final state and every corrected value. | https://www.tensortonic.com/problems/jax-l03-ema-with-scan |
| Spectral Norm by Power Iteration | Estimate the largest singular value of a matrix with power iteration inside jax.lax.fori_loop, using a traced iteration count. | https://www.tensortonic.com/problems/jax-l03-power-iteration-spectral-norm |
| Static Arguments in jit | Split token vectors into attention heads, RMS-normalize each head and merge them back under jax.jit with a static head count. | https://www.tensortonic.com/problems/jax-l03-static-arguments-jit |
| Traced Learning Rate Schedule | Compute linear warmup followed by cosine decay for a traced JAX step with jnp.where, for one step or a whole vector of steps. | https://www.tensortonic.com/problems/jax-l03-traced-lr-schedule |
| Batched Attention with vmap | Write scaled dot-product attention for one head in JAX and batch it over the batch and head axes with nested jax.vmap. | https://www.tensortonic.com/problems/jax-l04-batched-attention-vmap |
| Model Ensemble with in_axes | Evaluate a stacked ensemble of small MLPs on shared inputs in JAX with jax.vmap and in_axes, returning each prediction and the mean. | https://www.tensortonic.com/problems/jax-l04-ensemble-in-axes |
| Pairwise Distances with Nested vmap | Build pairwise squared distance and cosine similarity matrices in JAX by nesting jax.vmap over two point sets, with no loops. | https://www.tensortonic.com/problems/jax-l04-pairwise-distances-nested-vmap |
| Global Norm Gradient Clipping | Clip a JAX gradient pytree by its global L2 norm under jax.jit, with a float32 norm and one shared scale for every leaf. | https://www.tensortonic.com/problems/jax-l05-global-norm-clipping |
| SGD with Momentum over a Pytree | Apply one SGD with momentum step to a nested JAX parameter pytree with jax.tree.map, keeping its structure and dtypes. | https://www.tensortonic.com/problems/jax-l05-momentum-sgd-pytree |
| Implement Sigmoid in NumPy | Implement a vectorized sigmoid activation in NumPy for scalars, lists, vectors, and matrices, including large positive and negative inputs. | https://www.tensortonic.com/problems/sigmoid-numpy |

View my verified ML profile: [TensorTonic profile](https://www.tensortonic.com/profile/yhn90116)
<!-- tensortonic:end -->
