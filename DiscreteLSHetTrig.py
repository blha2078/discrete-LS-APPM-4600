import numpy as np
import matplotlib.pyplot as plt

def func(x):
    return 0.25 * x**2 + 3 * np.sin(x) - 2 * x + 5

def generate_heteroscedastic_data(x, objective_func, variance_func):
    """ Generate heteroscedastic data based on an x array, objective function, and variance function. """
    y_true = objective_func(x)
    variance = variance_func(x)
    noise = np.random.normal(0, np.sqrt(variance), len(x))
    y_observed = y_true + noise
    return y_true, y_observed, variance

def construct_design_matrix(x):
    """ Construct the design matrix including polynomial and trigonometric terms. """
    # Including polynomial terms up to x^2 and trigonometric terms
    X = np.column_stack((np.ones_like(x), x, x**2, np.sin(x), np.cos(x)))
    return X

def weighted_least_squares(X, y, weights=None):
    """ Perform weighted least squares fitting. """
    if weights is not None:
        W = np.diag(weights)
        X = W @ X  # Weighting the design matrix
        y = W @ y  # Weighting the response variable
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    return coeffs

def plot_results(x, y_true, y_observed, y_fit, variance, weights, variance_desc, weights_desc):
    """ Plot the results of the fitting along with variance and model error. """
    residuals = np.abs(y_true - y_fit)
    fig, ax = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    ax[0].scatter(x, y_observed, color='red', label='Noisy Data')
    ax[0].plot(x, y_fit, label='Fitted Curve', linestyle='--')
    ax[0].plot(x, y_true, label='True Function', color='green')
    ax[0].set_title('Fit to Noisy Data')
    ax[0].legend()
    ax[1].plot(x, variance, label='Variance', color='blue')
    ax[1].set_title('Variance Function: ' + variance_desc)
    ax[1].legend()
    ax[2].plot(x, residuals, label='Model Error', color='magenta')
    ax[2].set_title('Error of Model: ' + weights_desc)
    ax[2].legend()
    plt.show()

x_values = np.linspace(0, 10, 100)

# Different variance functions for demonstration
variance_functions = [
    (lambda x: 0.5 + 0.05*x**2, '0.5 + 0.05*x^2'),
    (lambda x: 1 + np.cos(x), '1 + cos(x)'),
]

for var_func, var_desc in variance_functions:
    y_true, y_observed, variance = generate_heteroscedastic_data(x_values, func, var_func)
    X = construct_design_matrix(x_values)
    for weight_func, weight_desc in weight_functions:
        weights = weight_func(variance)
        coefficients = weighted_least_squares(X, y_observed, weights)
        y_fit = X @ coefficients
        plot_results(x_values, y_true, y_observed, y_fit, variance, weights, var_desc, weight_desc)
