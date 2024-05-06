import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor

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
    X = np.column_stack((np.ones_like(x), x, x**2, np.sin(x), np.cos(x)))
    return X

def huber_fit(X, y):
    """ Fit the data using Huber regression. """
    huber = HuberRegressor()
    huber.fit(X, y)
    return huber.coef_, huber.intercept_

def plot_results(x, y_true, y_observed, y_fit, variance):
    """ Plot the results of the fitting along with variance and model error. """
    residuals = np.abs(y_true - y_fit)
    fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    ax[0].scatter(x, y_observed, color='red', label='Noisy Data')
    ax[0].plot(x, y_fit, label='Fitted Curve', linestyle='--')
    ax[0].plot(x, y_true, label='True Function', color='green')
    ax[0].set_title('Fit to Noisy Data Using Huber Loss')
    ax[0].legend()
    ax[1].plot(x, residuals, label='Model Error', color='magenta')
    ax[1].set_title('Error of Model')
    ax[1].legend()
    plt.show()

x_values = np.linspace(0, 10, 100)

# Variance function
var_func, var_desc = (lambda x: 0.5 + 0.05*x**2, '0.5 + 0.05*x^2')

# Generate data
y_true, y_observed, variance = generate_heteroscedastic_data(x_values, func, var_func)
X = construct_design_matrix(x_values)

# Perform Huber regression
coefficients, intercept = huber_fit(X, y_observed)
y_fit = X @ coefficients + intercept

# Plot results
plot_results(x_values, y_true, y_observed, y_fit, variance)
