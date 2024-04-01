import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared
from numpy import vstack, asarray, random, arange
from scipy.stats import norm
from numpy.random import normal
from numpy import argmax
from matplotlib.ticker import AutoLocator

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Charter"],
    "pgf.texsystem": "pdflatex"})

def objective(x, func_code, noise=0.1):
    """Objective function with optional noise."""
    noise = normal(loc=0, scale=noise)
    return eval(func_code)

def surrogate(model, X):
    """Surrogate or approximation for the objective function."""
    return model.predict(X, return_std=True)

def acquisition(X, Xsamples, model):
    """Probability of improvement acquisition function."""
    yhat, _ = surrogate(model, X)
    best = max(yhat)
    mu, std = surrogate(model, Xsamples)
    probs = norm.cdf((mu - best) / (std + 1E-9))
    return probs

def opt_acquisition(X, y, model, domain_min, domain_max):
    """Optimize the acquisition function."""
    Xsamples = np.random.uniform(domain_min, domain_max, 100)
    scores = acquisition(X, Xsamples.reshape(-1, 1), model)
    ix = np.argmax(scores)
    return Xsamples[ix]

def bayesian_optimization(func_code, domain_min, domain_max, initial_points=10, iterations=100, kernel_type='RBF', **kwargs):
    """Perform Bayesian Optimization."""
    X = np.random.uniform(domain_min, domain_max, initial_points)
    y = asarray([objective(x, func_code) for x in X])
    X = X.reshape(len(X), 1)
    y = y.reshape(len(y), 1)
    
    kernel = None
    if kernel_type == 'RBF':
        kernel = RBF(**kwargs)
    elif kernel_type == 'Matern':
        kernel = Matern(**kwargs)
    elif kernel_type == 'RationalQuadratic':
        kernel = RationalQuadratic(**kwargs)
    elif kernel_type == 'ExpSineSquared':
        kernel = ExpSineSquared(**kwargs)

    model = GaussianProcessRegressor(kernel=kernel)
    model.fit(X, y)
    
    print("Initial Surrogate:")
    plot(X, y, model, func_code, domain_min, domain_max)
    
    for i in range(iterations):
        x = opt_acquisition(X, y, model, domain_min, domain_max)
        actual = objective(x, func_code)
        est, _ = surrogate(model, [[x]])
        X = vstack((X, [[x]]))
        y = vstack((y, [[actual]]))
        model.fit(X, y)
    
    print("Final Surrogate")
    plot(X, y, model, func_code, domain_min, domain_max)
    best_ix = argmax(y)
    print(f'Best Result: x={X[best_ix][0]:.3f}, y={y[best_ix][0]:.3f}')

def plot(X, y, model, func_code, domain_min, domain_max):
    """Plot real observations vs surrogate function."""
    fig, ax = plt.subplots(figsize=(9, 3))
    
    # Plot the input function
    x_values = np.linspace(domain_min, domain_max, 1000)
    y_values = [objective(x, func_code, noise=0) for x in x_values]
    ax.plot(x_values, y_values, label='True Function', color='red', linestyle='--', zorder=1)

    # Plot the surrogate function
    ax.scatter(X, y, s=20, label='Samples', zorder=3, color='midnightblue')
    Xsamples = asarray(arange(min(X)[0], max(X)[0], 0.01))
    Xsamples = Xsamples.reshape(len(Xsamples), 1)
    ysamples, y_std = surrogate(model, Xsamples)
    ax.plot(Xsamples, ysamples, label='GP Mean', color='blue', zorder=2)
    ax.fill_between(Xsamples.ravel(), ysamples - 1.96 * y_std, ysamples + 1.96 * y_std,
                    alpha=0.2, color='blue', label = 'Confidence Interval')
    
    # Graph formatting
    ax.grid(alpha=0.4)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('$x$',fontsize=18)
    ax.set_ylabel('$f(x)$',fontsize=18)
    ax.xaxis.set_major_locator(AutoLocator())
    ax.yaxis.set_major_locator(AutoLocator())
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    ax.legend(loc='upper center', frameon=False, ncol=4,
              bbox_to_anchor=(0.5, 1.2),fontsize=14)
    plt.show()

def main():
    print('Bayesian Optimization')
    print('This program is a basic demonstration of Bayesian Optimization.')

    # Input function for BO
    print("\nFunction & Domain")
    func_code = input("Enter the function in Python format: ")
    domain_min = float(input("Enter the minimum value of the domain: "))
    domain_max = float(input("Enter the maximum value of the domain: "))

    # Set no. of initial samples and iterations
    initial_points = int(input("Enter the number of initial points: "))
    iterations = int(input("Enter the number of iterations: "))

    # Kernel selection
    print("\nKernel Selection")
    kernel_type = input("Select Kernel Type (RBF, Matern, RationalQuadratic, ExpSineSquared): ")

    # Additional hyperparameters based on the selected kernel type
    if kernel_type == 'Matern':
        kernel_length_scale = float(input("Enter Length Scale: "))
        kernel_nu = float(input("Enter Nu: "))
    elif kernel_type == 'RationalQuadratic':
        kernel_alpha = float(input("Enter Alpha: "))
        kernel_length_scale = float(input("Enter Length Scale: "))
    elif kernel_type == 'ExpSineSquared':
        kernel_length_scale = float(input("Enter Length Scale: "))
        kernel_periodicity = float(input("Enter Periodicity: "))
    else:  # RBF kernel by default
        kernel_length_scale = float(input("Enter Length Scale: "))

    kernel_params = {}
    if kernel_type == 'Matern':
        kernel_params['length_scale'] = kernel_length_scale
        kernel_params['nu'] = kernel_nu
    elif kernel_type == 'RationalQuadratic':
        kernel_params['alpha'] = kernel_alpha
        kernel_params['length_scale'] = kernel_length_scale
    elif kernel_type == 'ExpSineSquared':
        kernel_params['length_scale'] = kernel_length_scale
        kernel_params['periodicity'] = kernel_periodicity
    else:  # RBF kernel by default
        kernel_params['length_scale'] = kernel_length_scale

    bayesian_optimization(func_code, domain_min, domain_max, initial_points=initial_points,
                          iterations=iterations, kernel_type=kernel_type, **kernel_params)

if __name__ == '__main__':
    main()
