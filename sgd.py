import numpy as np
from timer import timer_func


@timer_func
def stochastic_gradient_descent(X, Y, deg_polynom=1, lr=0.001, max_epochs=100):
    weights = np.random.uniform(low=-0.5, high=0.5, size=deg_polynom+1)
    powers = np.array(range(deg_polynom+1))
    for _ in range(max_epochs):
        for x, y in zip(X, Y):
            h = np.sum(weights * x**powers)
            weights += lr*(y-h)*x**powers
    print(weights)
    return weights[::-1]


@timer_func
def stochastic_gradient_descent_list_comp_matrix(X, Y, deg_polynom=1, lr=0.001, max_epochs=100):
    weights = np.random.uniform(low=-0.5, high=0.5, size=deg_polynom+1)
    for _ in range(max_epochs):
        for x, y in zip(X, Y):
            h = sum([w * x**i for i, w in enumerate(weights)])
            weights = [w + lr*(y-h)*x**i for i, w in enumerate(weights)]
    print(weights)
    return weights[::-1]
