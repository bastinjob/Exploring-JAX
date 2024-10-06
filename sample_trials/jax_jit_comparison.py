import jax.numpy as jnp
from jax import jit
from jax import random
import timeit

def selu(x, alpha=1.67, lmbda=1.05):
  return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

'''
x = jnp.arange(5.0)
print(selu(x))
'''

key = random.key(1701)
x = random.normal(key, (1000000,))

exec_time = timeit.timeit(lambda: selu(x).block_until_ready(), number=10000)
print("Execution time without jit: " , exec_time)

selu_jit = jit(selu)
exec_time = timeit.timeit(lambda: selu_jit(x).block_until_ready(), number=10000)
print("Execution time for jit: " , exec_time)
