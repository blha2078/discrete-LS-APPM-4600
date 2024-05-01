import numpy as np
import matplotlib.pyplot as plt

def func(x):
    return np.sin(x)

def generate_heteroscedastic_data(x, objective_func, variance_func):
    """ Generate heteroscedastic data based on an x array, objective function, and variance function. """
    # Generate true y values based on the objective function
    y_true = objective_func(x)
    
    # Generate variance values using the variance function
    variance = variance_func(x)
    
    # Generate heteroscedastic noise
    noise = np.random.normal(0, np.sqrt(variance), len(x))
    
    # Add noise to y values to create observed values
    y_observed = y_true + noise
    
    return y_true, y_observed

def least_squares_poly(x, y, degree, weights=None, only_x_squared=False):
    """ Calculate polynomial regression coefficients using least squares with optional weights. """
    if only_x_squared:
        # Only use x^2 term for the regression
        X = np.vstack((x**2, np.ones_like(x))).T 
    else:
        # General polynomial fit
        X = np.vander(x, degree + 1)
    
    if weights is not None:
        W = np.diag(np.sqrt(weights))
        # Multiplying X and y by weights *** only step different ***
        X = np.dot(W, X)
        y = np.dot(W, y)

    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    return coeffs

def plot_regression(y_noisy, y_data, x_range, coeffs, label, ax1, ax2, ax3, only_x_squared=False):
    """ Plot regression curve along with the original data. """
    if only_x_squared:
        y_fit = coeffs[0] * x_range**2
    else:
        y_fit = np.polyval(coeffs, x_range)
        
    residuals_model = y_data - y_fit 
    residuals_noise = y_noisy - y_fit
    
    ax1.plot(x_range, y_fit, label=label, linestyle='--')
    ax1.grid(True)
    
    ax2.plot(x_range, np.abs(residuals_model), label=f'Error for {label}', linestyle='-')
    ax2.grid(True)
    
    ax3.plot(x_range, np.abs(residuals_noise), label=f'Error for {label}', linestyle='-')
    ax3.grid(True)

def perform_and_plot_heteroscedastic(x_start, x_end, use_weights=False):
    """ Generate heteroscedastic data, perform regression with or without weighted least squares, and plot the results. """
    x_noisy = np.linspace(x_start, x_end, 100)
    
    # Define your objective function
    objective_func = func
    
    # Define your variance function
    variance_func = lambda x: (x+0.01)**2
    
    # Generate heteroscedastic data
    y_true, y_observed = generate_heteroscedastic_data(x_noisy, objective_func, variance_func)
    x_range = np.linspace(x_start, x_end, 100)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 9), sharex=True)

    # Define weights
    # weights = np.linspace(1, 0.1, len(x_noisy))  # Example: Weight lower points more
    # Compute weights based on the variance function
    variance = variance_func(x_noisy)
    weights = np.sqrt(variance)  # Inverse of variance as weights
    print(weights)
        
    # Pure quadratic fit ax^2 with or without weighted least squares
    coeffs_quad_weighted = least_squares_poly(x_noisy, y_observed, 2, weights=weights, only_x_squared=True)
    plot_regression(y_observed, y_true, x_range, coeffs_quad_weighted, 'Weighted f1(x) = ax^2', ax1, ax2, ax3, only_x_squared=True)
    
    # # Quadratic fit ax^2 + bx + c with or without weighted least squares
    # coeffs_quad_lin_weighted = least_squares_poly(x_noisy, y_observed, 2, weights=weights)
    # plot_regression(y_observed, y_true, x_range, coeffs_quad_lin_weighted, 'Weighted f2(x) = ax^2 + bx + c', ax1, ax2, ax3)

    # # 4th degree polynomial fit ax^4 + bx^3 + cx^2 + dx + e with or without weighted least squares
    # coeffs_degree4_weighted = least_squares_poly(x_noisy, y_observed, 4, weights=weights)
    # plot_regression(y_observed, y_true, x_range, coeffs_degree4_weighted, 'Weighted f3(x) = ax^4 + bx^3 + cx^2 + dx + e', ax1, ax2, ax3)

    fig.suptitle('Weights used: {}'.format('Yes' if use_weights else 'No'))

    ax1.scatter(x_noisy, y_observed, color='red', label='Noisy Data')
    ax1.plot(x_range, func(x_range), label='True Function')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Polynomial Regression Fit')
    ax1.legend()
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Error for Model')
    ax2.legend()
    
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Error for Noise')
    ax3.legend()
    
    plt.show()

# Perform regression with weighted least squares on the interval [0, 5]
perform_and_plot_heteroscedastic(0, 5, use_weights=True)

# Perform regression without weighted least squares on the interval [0, 5]
perform_and_plot_heteroscedastic(0, 5, use_weights=False)
