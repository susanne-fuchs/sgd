import pandas as pd
from data import generate_data_points, plot_data
from sgd import stochastic_gradient_descent, stochastic_gradient_descent_list_comp_matrix

if __name__ == "__main__":
   # df = generate_data_points(100, "data/data.csv")
   # plot_data("data/data.csv")

   lr = 0.1
   epochs = 5000
   deg = 3

   df = pd.read_csv("data/data.csv")
   weights = stochastic_gradient_descent(df.x, df.y, deg, lr, epochs)
   plot_data("data/data.csv", weights)

   weights = stochastic_gradient_descent_list_comp_matrix(df.x, df.y, deg, lr, epochs)
   plot_data("data/data.csv", weights)