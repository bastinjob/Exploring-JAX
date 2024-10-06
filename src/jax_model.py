import jax.numpy as jnp
from jax import random, jit
import utils  # Assuming utils.py is in the same directory

# Define the model
def model(x):
    return jnp.dot(x, x.T)  # Example model, modify as needed

@utils.measure_time
def train_jax(x, epochs=1000):
    for epoch in range(epochs):
        _ = model(x)
        if epoch % (epochs // 10) == 0:  # Log every 10% of epochs
            print(f"Epoch: {epoch}")
    
@utils.measure_time
def train_jax_jit(x, epochs=1000):
    model_jit = jit(model)
    for epoch in range(epochs):
        _ = model_jit(x)
        if epoch % (epochs // 10) == 0:  # Log every 10% of epochs
            print(f"Epoch: {epoch}")

# Main execution
if __name__ == "__main__":
    key = random.PRNGKey(1701)
    
    # Increase dataset size
    x = random.normal(key, (10_000, 10_000))  # Use a larger dataset

    print("Training without JIT:")
    utils.track_memory()  # Track memory before training
    train_jax(x, epochs=10)  # Increase epochs

    print("\nTraining with JIT:")
    utils.track_memory()  # Track memory before training
    train_jax_jit(x, epochs=10)  # Increase epochs
