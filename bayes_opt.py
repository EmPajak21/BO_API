import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal, random
from numpy import vstack, asarray
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import simplefilter


def objective(x, noise=0.1):
    """
    Objective function with optional noise.

    Parameters:
    - x (float): Input value.
    - noise (float): Optional noise level (default: 0.1).

    Returns:
    - float: Objective function value.
    """
    noise = normal(loc=0, scale=noise)
    return -(x**2) + 2 * x + 1 + noise


def surrogate(model, X):
    """
    Surrogate or approximation for the objective function.

    Parameters:
    - model (GaussianProcessRegressor): Surrogate model.
    - X (ndarray): Input values.

    Returns:
    - tuple: Predicted values and standard deviations.
    """
    simplefilter("ignore")
    return model.predict(X, return_std=True)


def acquisition(X, Xsamples, model):
    """
    Probability of improvement acquisition function.

    Parameters:
    - X (ndarray): Input values.
    - Xsamples (ndarray): Sampled input values.
    - model (GaussianProcessRegressor): Surrogate model.

    Returns:
    - ndarray: Probability of improvement values.
    """
    yhat, _ = surrogate(model, X)
    best = max(yhat)
    mu, std = surrogate(model, Xsamples)
    probs = norm.cdf((mu - best) / (std + 1e-9))
    return probs


def opt_acquisition(X, y, model):
    """
    Optimize the acquisition function.

    Parameters:
    - X (ndarray): Input values.
    - y (ndarray): Objective function values.
    - model (GaussianProcessRegressor): Surrogate model.

    Returns:
    - float: Optimal input value.
    """
    Xsamples = random(100) * (max(X) - min(X)) + min(X)
    Xsamples = Xsamples.reshape(len(Xsamples), 1)
    scores = acquisition(X, Xsamples, model)
    ix = np.argmax(scores)
    return Xsamples[ix, 0]


def bayesian_optimization(f=objective, domain=(0, 1), initial_points=10, iterations=100):
    """
    Perform Bayesian Optimization.

    Parameters:
    - f (function): Objective function.
    - domain (tuple): Domain of the input values (default: (0, 1)).
    - initial_points (int): Number of initial random points (default: 10).
    - iterations (int): Number of optimization iterations (default: 100).
    """
    X = random(initial_points) * (domain[1] - domain[0]) + domain[0]
    y = asarray([f(x) for x in X])
    X = X.reshape(len(X), 1)
    y = y.reshape(len(y), 1)

    model = GaussianProcessRegressor()
    model.fit(X, y)

    print("Initial Surrogate Function:")
    plot(X, y, model, f, domain)

    for i in range(iterations):
        x = opt_acquisition(X, y, model)
        actual = f(x)
        est, _ = surrogate(model, [[x]])
        X = vstack((X, [[x]]))
        y = vstack((y, [[actual]]))
        model.fit(X, y)

    print("Final Surrogate Function with Optimal Solution:")
    plot(X, y, model, f, domain)
    best_ix = np.argmax(y)
    print(f"Best Result: x={X[best_ix][0]:.3f}, y={y[best_ix][0]:.3f}")


def plot(X, y, model, f, domain):
    """
    Plot real observations vs surrogate function.

    Parameters:
    - X (ndarray): Input values.
    - y (ndarray): Objective function values.
    - model (GaussianProcessRegressor): Surrogate model.
    - f (function): Objective function.
    - domain (tuple): Domain of the input values.
    """
    plt.figure(figsize=(10, 4))
    plt.scatter(X, y, s=10, label="Samples")

    # Plot real function without noise
    X_real = np.linspace(domain[0], domain[1], 1000)
    y_real = [f(x, noise=0) for x in X_real]
    plt.plot(X_real, y_real, label="True Function (No Noise)", color="green")

    # Plot surrogate function and confidence intervals
    Xsamples = np.linspace(domain[0], domain[1], 1000).reshape(-1, 1)
    ysamples, y_std = surrogate(model, Xsamples)
    plt.plot(Xsamples, ysamples, label="Surrogate", color="red")
    plt.fill_between(
        Xsamples.ravel(),
        ysamples - 1.96 * y_std,
        ysamples + 1.96 * y_std,
        alpha=0.2,
        color="red",
    )
    plt.ylim(min(y_real) - 0.5, max(y_real) + 0.5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Surrogate Function with Actual Function (No Noise)")
    plt.legend()
    plt.xlim(domain[0], domain[1])
    plt.show()


# Example usage:
bayesian_optimization(f=objective, domain=(-3, 3), initial_points=5, iterations=10)
