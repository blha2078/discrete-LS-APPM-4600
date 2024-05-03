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

def least_squares_poly(x, y, degree, weights=None):
    """ Calculate polynomial regression coefficients using least squares with optional weights. """
    X = np.vander(x, degree + 1)
    if weights is not None:
        W = np.diag(weights)
        X = np.dot(W, X)
        y = np.dot(W, y)
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    return coeffs

def plot_results(x, y_true, y_observed, variance, weights, variance_desc, weights_desc):
    coeffs = least_squares_poly(x, y_observed, 2, weights=weights)
    y_fit = np.polyval(coeffs, x)
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
variance_functions = [
    (lambda x: 0.5 + 0.05*x**2, '0.5 + 0.05*x^2'),
    (lambda x: 1 + np.cos(x), '1 + cos(x)'),
]
weight_functions = [
    (lambda var: 1/var, 'Inverse of Variance'),
    (lambda var: 1/np.sqrt(var), 'Inverse of Square Root of Variance'),
]

for var_func, var_desc in variance_functions:
    y_true, y_observed, variance = generate_heteroscedastic_data(x_values, func, var_func)
    for weight_func, weight_desc in weight_functions:
        weights = weight_func(variance)
        plot_results(x_values, y_true, y_observed, variance, weights, var_desc, weight_desc)
