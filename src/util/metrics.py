import time
from statistics import mean
metrics = {}


def capture_timing_info(metrics_container=metrics):
    def decorator(func):
        def wrapper(*args, **kwargs):
            t0 = time.time()
            result = func(*args, **kwargs)
            t1 = time.time()
            if func.__name__ not in metrics_container:
                metrics_container[func.__name__] = [t1 - t0]
            else:
                metrics_container[func.__name__].append(t1 - t0)
            return result
        return wrapper
    return decorator


def print_metrics(verbose=False):
    for k, v in metrics.items():
        print(f"[{k}]")
        print(f"\t{'average:':<15}{mean(v):.8f} seconds")
        print(f"\t{'total:':<15}{sum(v):.8f} seconds")
        print(f"\t{'last:':<15}{v[-1]:.8f} seconds")
        if verbose:
            print(f"\t{'all captures:':<15}{v}")
