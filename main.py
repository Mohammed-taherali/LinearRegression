# Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

# Stylize the plots
style.use("ggplot")

# Generate the data
X, y = make_regression(n_samples=10000, n_features=1, noise=16)
y = y.reshape((-1, 1))

# Making X as a 2-D array (two independent variables)
X2 = X - np.random.uniform(0.2, 0.5, size=X.shape)
X = np.hstack((X2, X))


class MyLinearRegression:
    def __init__(self):
        self.cost = []


    def predict_y(self, X: np.ndarray):
  
        # multiply the theta and X values to find the expected value of y.
        prediction = np.dot(X, self.theta)
        
        return prediction


    def derivative_theta(self, x: np.ndarray, y: np.ndarray, y_pred: np.ndarray):
    
        # Find the value of gradient
        derivative = np.dot(x.transpose(), (y_pred - y))
        
        return derivative


    def update_cost(self, y: np.ndarray, y_pred: np.ndarray):

        cost_val = np.sum((y - y_pred) ** 2) / self.batch_size
        self.cost.append(cost_val)


    def update_coeff(self, x: np.ndarray, y: np.ndarray, y_pred: np.ndarray):
        
        # update theta
        theta = self.theta - self.learning_rate * self.derivative_theta(x, y, y_pred)
        return theta


    def generate_batches(self, X: np.ndarray, y: np.ndarray):

        data = np.hstack((X, y))
        np.random.shuffle(data)
        X, y = np.split(data, [-1, ], axis=1)
        batches = []

        for i in range(0, X.shape[0], self.batch_size):
            x_small = X[i: i + self.batch_size, :]
            y_small = y[i: i + self.batch_size, :]

            batches.append((x_small, y_small))

        return batches


    def score(self, X_test: np.ndarray, Y_test: np.ndarray):

        # Get predicted value of y
        y_pred = self.predict_y(X_test)
        
        # Find the R^2 value (the higher the better)
        r_sqr = 1 - (np.sum((Y_test - y_pred) ** 2) / np.sum((Y_test - Y_test.mean()) ** 2))
        return r_sqr


    def train_test_split(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.20):

        # Combine X (independent) and y (dependent) variables for splitting.
        data = np.hstack((X, y))
        data_size = data.shape[0]
        np.random.shuffle(data)

        # Split data again into X and y after shuffling.
        x, y = np.split(data, [-1, ], axis=1)

        # split into train and test parts
        X_train = x[: int((1 - test_size) * data_size)]
        X_test = x[int((1 - test_size) * data_size):]
        y_train = y[:int((1 - test_size) * data_size)]
        y_test = y[int((1 - test_size) * data_size):]

        return X_train, X_test, y_train, y_test


    def fit(self, X: np.ndarray, Y: np.ndarray, learning_rate: float = 0.001, batch_size: int = 32):

        # Initialize the values of all thetas to 0
        self.theta = np.zeros((X.shape[1], 1))
        max_itr = 10
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        for _ in range(max_itr):
            # Get batches
            batches = self.generate_batches(X, Y)
            for batch in batches:
                x, y = batch
                
                # Get the predicted values of y
                y_pred = self.predict_y(x)
                
                # Update the values of theta and cost
                self.theta = self.update_coeff(x, y, y_pred)
                self.update_cost(y, y_pred)


if __name__ == "__main__":

    # Instantiate the class.
    reg = MyLinearRegression()

    # Split the data.
    X_train, X_test, y_train, y_test = reg.train_test_split(X, y)

    # Fitting the data.
    reg.fit(X_train, y_train)

    # Get value of R^2.
    score = reg.score(X_test, y_test)
    print(f"Our R^2 value is {score:.4f}.")


    # Sklearn's LinearRegression class.
    rg = LinearRegression()
    rg.fit(X_train, y_train)
    sk_scr = rg.score(X_test, y_test)
    print(f"Sklearn's score {sk_scr:.4f}.")

    # Make predictions.
    y_prd = reg.predict_y(X_test)

    # Plotting the data.
    fig, ax = plt.subplots(figsize=(9, 5))
    plt.plot(X_test[:, 1], y_prd, color='b', label="Y predicted")
    plt.scatter(X_test[:, 1], y_test, s=10, label="Y actual")
    plt.title("Plotting the predicted value of Y")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    fig.savefig("y_actual_vs_y_pred.png")
    plt.show()

    # Plotting the error.
    fig, ax = plt.subplots(figsize=(9, 5))
    plt.plot(reg.cost)
    plt.title("Cost function V/S No. of iterations")
    plt.xlabel("No. of iterations.")
    plt.ylabel("Cost value")
    fig.savefig("cost.png")
    plt.show()
