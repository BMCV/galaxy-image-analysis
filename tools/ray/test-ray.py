import numpy as np
import ray


@ray.remote
def f(y, x):
    return y * x


if __name__ == "__main__":

    np.random.seed(0)
    y = np.random.rand(1024, 1024)

    ray.init(num_cpus=1, log_to_driver=True)

    y_id = ray.put(y)
    futures = [f.remote(y_id, x) for x in range(4)]
    results = [ray.get(future) for future in futures]

    for result in results:
        print(result.mean())
