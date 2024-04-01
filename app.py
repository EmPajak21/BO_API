import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared
from numpy import vstack, asarray, random, arange
from scipy.stats import norm
from numpy.random import normal
from numpy import argmax
from matplotlib.ticker import AutoLocator

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
    
    st.subheader("Initial Surrogate:")
    plot(X, y, model, func_code, domain_min, domain_max)
    
    for i in range(iterations):
        x = opt_acquisition(X, y, model, domain_min, domain_max)
        actual = objective(x, func_code)
        est, _ = surrogate(model, [[x]])
        X = vstack((X, [[x]]))
        y = vstack((y, [[actual]]))
        model.fit(X, y)
    
    st.subheader("Final Surrogate")
    plot(X, y, model, func_code, domain_min, domain_max)
    best_ix = argmax(y)
    #st.write(f'Best Result: x={X[best_ix][0]:.3f}, y={y[best_ix][0]:.3f}')

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
    # Save the graph as SVG
    plt.savefig('graph.svg', format='svg')
    st.pyplot(fig)

def main():
    st.title('Bayesian Optimization API')
    st.write('This API is a basic interactive demonstration of Bayesian Optimization.')

    # Input function for BO
    st.subheader("Function & Domain")
    func_code = st.text_area("Python format", 
                value='np.sin(x) + np.cos(2*x) + np.exp(-0.5 * (x - 5)**2) - 0.5 * np.sin(3*x)')

    # Function domain input
    col1, col2 = st.columns(2)
    with col1:
        domain_min = st.number_input('Minimum Value', value=-5.0)
    with col2:
        domain_max = st.number_input('Maximum Value', value=5.0)

    # Set no. of initial samples and iterations
    initial_points = st.slider('Number of Initial Points', min_value=1, max_value=20, value=5)
    iterations = st.slider('Number of Iterations', min_value=1, max_value=20, value=5)

    # Kernel selection
    st.subheader('Kernel Selection')
    col1, col2 = st.columns(2)
    with col1:
        kernel_type = st.radio('Select Kernel Type', ('RBF', 'Matern', 'RationalQuadratic', 'ExpSineSquared'))

    # Additional hyperparameters based on the selected kernel type
    with col2:
        if kernel_type == 'Matern':
            kernel_length_scale = st.number_input('Length Scale', value=1.0)
            kernel_nu = st.number_input('Nu', value=2.5)
        elif kernel_type == 'RationalQuadratic':
            kernel_alpha = st.number_input('Alpha', value=1.0)
            kernel_length_scale = st.number_input('Length Scale', value=1.0)
        elif kernel_type == 'ExpSineSquared':
            kernel_length_scale = st.number_input('Length Scale', value=1.0)
            kernel_periodicity = st.number_input('Periodicity', value=1.0)
        else:  # RBF kernel by default
            kernel_length_scale = st.number_input('Length Scale', value=1.0)

    # Further options for hyperparameters depending on kernel
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

    if st.button('Run Bayesian Optimization'):
        bayesian_optimization(func_code, domain_min, domain_max, initial_points=initial_points,
                              iterations=iterations, kernel_type=kernel_type, **kernel_params)

if __name__ == '__main__':
    main()
