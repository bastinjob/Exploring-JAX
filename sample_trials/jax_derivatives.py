from jax import grad
import jax.numpy as jnp
from jax import jit


def sum_logistic(x):
    return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))

x_small = jnp.arange(3.0)
derivative_fn = grad(sum_logistic)
print(derivative_fn(x_small))

def first_finite_differences(f, x, eps=1E-3):
  return jnp.array([(f(x + eps * v) - f(x - eps * v)) / (2 * eps)
                   for v in jnp.eye(len(x))])

print(first_finite_differences(sum_logistic, x_small))

print(grad(jit(grad(jit(grad(sum_logistic)))))(1.0))

