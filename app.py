import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.gaussian_process import GaussianProcessRegressor
from numpy import vstack, asarray, random, arange
from scipy.stats import norm
from numpy.random import normal
from numpy import argmax
import matplotlib.pyplot as plt


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

def bayesian_optimization(func_code, domain_min, domain_max, initial_points=10, iterations=100):
    """Perform Bayesian Optimization."""
    X = np.random.uniform(domain_min, domain_max, initial_points)
    y = asarray([objective(x, func_code) for x in X])
    X = X.reshape(len(X), 1)
    y = y.reshape(len(y), 1)
    
    model = GaussianProcessRegressor()
    model.fit(X, y)
    
    st.header("Initial Surrogate Function:")
    plot(X, y, model, func_code, domain_min, domain_max)
    
    for i in range(iterations):
        x = opt_acquisition(X, y, model, domain_min, domain_max)
        actual = objective(x, func_code)
        est, _ = surrogate(model, [[x]])
        X = vstack((X, [[x]]))
        y = vstack((y, [[actual]]))
        model.fit(X, y)
    
    st.header("Final Surrogate Function")
    plot(X, y, model, func_code, domain_min, domain_max)
    best_ix = argmax(y)
    #st.write(f'Best Result: x={X[best_ix][0]:.3f}, y={y[best_ix][0]:.3f}')

def plot(X, y, model, func_code, domain_min, domain_max):
    """Plot real observations vs surrogate function."""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Plot the input function
    x_values = np.linspace(domain_min, domain_max, 1000)
    y_values = [objective(x, func_code, noise=0) for x in x_values]
    ax.plot(x_values, y_values, label='Input Function', color='blue')

    # Plot the surrogate function
    ax.scatter(X, y, s=50, label='Samples')
    Xsamples = asarray(arange(min(X), max(X), 0.01))
    Xsamples = Xsamples.reshape(len(Xsamples), 1)
    ysamples, y_std = surrogate(model, Xsamples)
    ax.plot(Xsamples, ysamples, label='Surrogate Function', color='red')
    ax.fill_between(Xsamples.ravel(), ysamples - 1.96 * y_std, ysamples + 1.96 * y_std, alpha=0.2, color='red')
    ax.grid(alpha=0.4)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend(loc='upper center',frameon=False)

    st.pyplot(fig)

def main():
    st.title('Bayesian Optimization API')
    st.write('This API is an basic interactive demonstration of Bayesian Optimization.')

    st.header("Objective Function")
    func_code = st.text_area("Python format", value='x**2 - 0.5*x + 1')

    st.header("Domain of Objective Function")
    domain_min = st.number_input('Minimum Value', value=-10.0)
    domain_max = st.number_input('Maximum Value', value=10.0)

    initial_points = st.slider('Number of Initial Points', min_value=1, max_value=20, value=5)
    iterations = st.slider('Number of Iterations', min_value=1, max_value=50, value=10)

    if st.button('Run Bayesian Optimization'):
        bayesian_optimization(func_code, domain_min, domain_max, initial_points=initial_points, iterations=iterations)

if __name__ == '__main__':
    main()
