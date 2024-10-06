import time
import tracemalloc

def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Time taken: {end - start:.4f} seconds")
        return result
    return wrapper

def track_memory():
    tracemalloc.start()
    # Your function here
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 10**6}MB; Peak was {peak / 10**6}MB")
    tracemalloc.stop()
