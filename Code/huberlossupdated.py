import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
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
    huber = HuberRegressor(epsilon=1)
    huber.fit(X, y)
    return huber.coef_, huber.intercept_

def ols_fit(X, y):
    """ Fit the data using Ordinary Least Squares (OLS) regression. """
    ols = LinearRegression()
    ols.fit(X, y)
    return ols.coef_, ols.intercept_

def plot_results(x, y_true, y_observed, y_fit_huber, y_fit_ols, variance):
    """ Plot the results of both Huber and OLS fitting along with variance and model error. """
    residuals_huber = np.abs(y_true - y_fit_huber)
    residuals_ols = np.abs(y_true - y_fit_ols)
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    ax[0].scatter(x, y_observed, color='red', label='Noisy Data')
    ax[0].plot(x, y_fit_huber, label='Huber Fitted Curve', linestyle='--')
    ax[0].plot(x, y_fit_ols, label='OLS Fitted Curve', linestyle='-.')
    ax[0].plot(x, y_true, label='True Function', color='green')
    ax[0].set_title('Fit to Noisy Data Using Huber Loss vs. OLS')
    ax[0].legend()
    
    ax[1].plot(x, residuals_huber, label='Huber Model Error', color='magenta')
    ax[1].plot(x, residuals_ols, label='OLS Model Error', color='blue')
    ax[1].set_title('Error of Huber vs. OLS Models')
    ax[1].legend()
    
    plt.show()

# Generate data with bad outliers
np.random.seed(1)
x_values = np.linspace(0, 10, 100)
y_true, y_observed, variance = generate_heteroscedastic_data(x_values, func, lambda x: 0.5 + 0.05 * x**2)

# Introduce bad outliers
outlier_indices = np.random.choice(len(x_values), size=5, replace=False)
y_observed[outlier_indices] += np.random.normal(loc=50, scale=20, size=5)

# Construct design matrix
X = construct_design_matrix(x_values)

# Perform Huber regression
coefficients_huber, intercept_huber = huber_fit(X, y_observed)
y_fit_huber = X @ coefficients_huber + intercept_huber

# Perform OLS regression
coefficients_ols, intercept_ols = ols_fit(X, y_observed)
y_fit_ols = X @ coefficients_ols + intercept_ols

# Plot results
plot_results(x_values, y_true, y_observed, y_fit_huber, y_fit_ols, variance)
