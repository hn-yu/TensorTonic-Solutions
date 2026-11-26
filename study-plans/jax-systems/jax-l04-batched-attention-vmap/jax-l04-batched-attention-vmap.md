# Batched Attention with vmap

Attention layers run on arrays with two axes that have nothing to do with the attention rule itself, the batch axis and the head axis. Writing the rule with those axes in place means every matrix product needs an einsum string or a transpose of the last two axes, and a slip in either one gives a silently wrong answer. With JAX you can instead write attention for one head of one example, where it reads exactly like the formula, and let the transformation jax.vmap add the batch and head axes for you. This page explains what vmap does, builds attention for one example, and then stacks two vmaps to cover the batch and the heads.

## A function for one example

Most mathematical descriptions of a layer talk about one example. The attention formula multiplies a query matrix by a transposed key matrix, and a transpose has one obvious meaning only for a two-dimensional matrix. Real inputs arrive as four-dimensional arrays of shape (B, H, Tq, D), where B is the batch size, H is the number of heads, Tq is the number of queries and D is the width of each query vector.

Code written against the full four-dimensional shape has to say which axes a product contracts and which axes it carries along. That bookkeeping is where bugs hide. Transposing every axis of a four-dimensional array reverses their order, so an expression that is correct for a matrix becomes wrong for a batch. A function that only ever sees one (Tq, D) block avoids the problem, because there is only one way to transpose a matrix.

## What vmap does to a function

The transformation jax.vmap takes a function written for single examples and returns a new function that accepts a stack of examples along a leading axis. The new function behaves as if it sliced the inputs along that axis, called the original function on each slice and stacked the results along a new axis 0. In symbols, if $f$ maps an input $x$ to an output $y$, the vmapped function maps a stack $X$ with rows $x_1, \dots, x_n$ to the stack of outputs $f(x_1), \dots, f(x_n)$.

Take a tiny function that returns the dot product of a vector of length 3 with the fixed weight vector $w = (1, -1, 2)$. For the vector $(1, 2, 3)$ it returns $1 - 2 + 6 = 5$, and for $(0, 1, 0)$ it returns $-1$. Applying the vmapped version to the (2, 3) matrix with those two rows returns the array $(5, -1)$ in one call.

Instead of running a Python loop over the rows, JAX traces the function once with abstract placeholder values, called tracers, that know their shape and dtype but not their contents. Every primitive operation the function performs, such as a dot product, an exponential or a sum, has a batching rule that says how to apply it to an extra leading axis. In the dot product example the traced program contains a single matrix-vector product of the (2, 3) matrix with $w$, the same operation you would have written by hand. A vmapped function therefore costs about as much as carefully hand-batched code.

## Attention for one head

For one head of one example, write $Q$ for the (Tq, D) block of queries, $K$ for the (Tk, D) block of keys and $V$ for the (Tk, Dv) block of values, where Tk is the number of keys and Dv is the width of each value vector. The output is

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left(\frac{Q K^{\top}}{\sqrt{D}}\right) V .
$$

The product $Q K^{\top}$ has shape (Tq, Tk), and its entry in row $i$ and column $j$ is the dot product of query $i$ with key $j$, a measure of how well they match. The softmax turns each row into weights that are positive and sum to 1, and multiplying by $V$ averages the value vectors with those weights. The output has shape (Tq, Dv), one averaged value vector per query.

Take one query $q = (3, 1)$, two keys $(1, 0)$ and $(0, 1)$, and two one-dimensional values 10 and 20. The dot products are 3 and 1, and dividing by $\sqrt{2}$ gives scores of 2.1213 and 0.7071. The softmax turns these into weights 0.8044 and 0.1956, so the output is $0.8044 \cdot 10 + 0.1956 \cdot 20 = 11.9557$. The first key matches the query better, so its value receives about four fifths of the weight.

## Why the scores are divided by the square root of D

A dot product of two vectors of width D adds up D products. If every entry is an independent number with mean 0 and variance 1, each product also has variance 1, so the sum has variance D and standard deviation $\sqrt{D}$. With D = 4 a typical score is about 2 in size, and with D = 64 it is about 8.

Large scores make the softmax extremely peaked. The scores (1, 0.5, 0) give weights of about 0.506, 0.307 and 0.186, while the scores (8, 4, 0) give 0.982, 0.018 and 0.0003. When almost all the weight sits on one key, the gradients with respect to the other scores are close to zero and the layer learns slowly. Dividing by $\sqrt{D}$ brings the typical score back to a size near 1 whatever the width, which is why Vaswani et al. (2017) called the operation scaled dot-product attention.

## Keeping the softmax finite

The softmax of a row of scores $s_1, \dots, s_n$ gives entry $j$ the value $e^{s_j}$ divided by the sum of all $e^{s_k}$. In float32 the exponential overflows to infinity once its input passes about 88.72, the natural logarithm of the largest float32 number. Applying the formula directly to the scores (100, 99) divides infinity by infinity, which gives NaN.

Subtracting the largest score from every score before exponentiating fixes this without changing the answer, because the same factor $e^{-\max}$ appears in the numerator and the denominator and cancels. The shifted scores are (0, -1), and the weights come out as 0.7311 and 0.2689. The function jax.nn.softmax performs this shift internally.

## Two vmaps for two axes

One vmap strips one leading axis, and the attention inputs have two leading axes to strip, the batch axis and the head axis, so the per-head function is wrapped twice. The wrapper applied last is the outermost one, and it maps over axis 0 of the full arrays, which is the batch axis. The function it maps is itself vmapped, so each of its calls receives arrays of shape (H, Tq, D), (H, Tk, D) and (H, Tk, Dv) and maps over their axis 0, which is now the head axis. The innermost function finally sees one head of one example.

With B = 2, H = 3, Tq = 4, Tk = 5, D = 8 and Dv = 6, the innermost function sees blocks of shape (4, 8), (5, 8) and (5, 6) and returns a (4, 6) block. The inner vmap stacks three of those into (3, 4, 6), and the outer vmap stacks two of those into (2, 3, 4, 6). Each vmap places its new axis at position 0 of its output by default, so the result comes back in the order (B, H, Tq, Dv) without any transpose.

By default vmap maps axis 0 of every argument, which suits attention because q, k and v all carry the batch and head axes in the same positions. The next problem uses the in_axes argument for inputs that should not be mapped at all.

## Tracing rules inside a vmapped function

Inside the per-head function the arguments are tracers, so their shapes are ordinary Python integers while their values are unknown. Reading the shape, for example to get D for the scale, is fine. Converting a value to a Python number, passing it to NumPy or using it in an if statement fails, because the trace has no concrete number to give. These are the same rules jax.jit imposes, and functions built with vmap compose freely with jax.jit, so the batched attention can be compiled as one program.

The dtype travels through the computation as well. If q, k and v are bfloat16, the products, the softmax and the weighted sum all stay in bfloat16. A Python number used as the scale keeps that dtype, while an explicit float32 array promotes every later result to float32.

## Common mistakes to avoid

- **Transposing the full array.** A plain transpose of a four-dimensional array reverses every axis, while inside the per-head function the same transpose correctly swaps the two axes of one matrix.
- **Dividing by the square root of the value width.** The scale uses D, the width shared by queries and keys, which differs from Dv whenever the values have another width.
- **Normalizing over the queries.** The softmax must run over the keys, the last axis of the (Tq, Tk) score matrix, so that each query's weights sum to 1.
- **Exponentiating raw scores.** Large scores overflow float32 and produce NaN unless the row maximum is subtracted first.
- **Promoting the dtype by accident.** A float32 array used as the scale turns bfloat16 scores into float32, so the output no longer has the dtype of q.
- **Branching on values inside the mapped function.** A Python if statement or a float conversion on a tracer fails under both vmap and jax.jit.