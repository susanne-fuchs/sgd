import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def generate_data_points(size, output_path=None):
    epsilons = np.random.uniform(low=-0.3, high=0.3, size=size)
    xs = np.random.uniform(low=0.0, high=1.0, size=size)
    ys = np.sin(2*np.pi*xs) + epsilons

    df = pd.DataFrame({"x": xs, "y": ys})
    print(df)

    if output_path:
        path, file = os.path.split(output_path)
        if not os.path.isdir(path):
            os.makedirs(path)
        df.to_csv(output_path, index=False)
    return df


def plot_data(file_path, weights=None):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = generate_data_points(100, file_path)

    plt.scatter(x=df.x, y=df.y, c="lightcoral")
    plt.xlabel("x")
    plt.ylabel("y")
    if weights is not None:
        xlin = np.linspace(0, 1, num=50)
        polynom = np.poly1d(weights)
        plt.plot(xlin, polynom(xlin), c="red")

    plt.show()
