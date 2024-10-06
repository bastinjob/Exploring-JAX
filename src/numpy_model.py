import numpy as np
from utils import measure_time, track_memory  # Importing the required functions

# Define linear regression function
def model(w, b, x):
    return np.dot(x, w) + b

# Define the loss function (mean squared error)
def loss(w, b, x, y):
    predictions = model(w, b, x)
    return np.mean((predictions - y) ** 2)

def grad_loss(w, b, x, y):
    N = x.shape[0]
    predictions = model(w, b, x)
    error = predictions - y
    dw = (1 / N) * np.dot(x.T, error)
    db = (1 / N) * np.sum(error)
    return dw, db

@measure_time  # Measure time for training
def train_step(w, b, x, y, learning_rate=0.01):
    dw, db = grad_loss(w, b, x, y)
    w = w - learning_rate * dw
    b = b - learning_rate * db
    return w, b

if __name__ == "__main__":
    # Initialize data (random data for illustration)
    np.random.seed(42)
    x = np.random.normal(size=(1000, 3))  # 1000 samples, 3 features
    true_w = np.array([2.0, -1.0, 0.5])
    true_b = 0.1
    y = np.dot(x, true_w) + true_b

    # Initialize weights and bias
    w = np.zeros(3)
    b = 0.0

    # Start memory tracking
    track_memory()

    # Training process
    epochs = 500
    for epoch in range(epochs):
        w, b = train_step(w, b, x, y)

    # Stop memory tracking and display final memory usage
    track_memory()

    # Print final parameters
    print(f"Final weights: {w}")
    print(f"Final bias: {b}")
