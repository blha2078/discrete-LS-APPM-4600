import numpy as np
import matplotlib.pyplot as plt 

def func(x):
    return x**2

def least_squares_poly(x, y, degree, only_x_squared=False):
    """ Calculate polynomial regression coefficients using least squares. """
    if only_x_squared:
        # Only use x^2 term for the regression
       X = np.vstack((x**2,np.ones_like(x))).T 
    else:
        # General polynomial fit
        X = np.vander(x, degree + 1)
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    print(coeffs)
    return coeffs

def plot_regression(x_data, y_data, x_range, coeffs, label, only_x_squared=False):
    """ Plot regression curve along with the original data. """
    if only_x_squared:
        y_fit = coeffs[0]*x_range**2
    else:
        y_fit = np.polyval(coeffs, x_range)
    plt.plot(x_range, y_fit, label=label, linestyle='--')
    plt.grid(True)

def perform_and_plot(x_start, x_end):
    x_noisy = np.linspace(x_start, x_end, 50)
    noise = np.random.normal(loc=0, scale=7, size=len(x_noisy))
    y_noisy = func(x_noisy) + noise
    x_range = np.linspace(x_start, x_end, 100)

    plt.figure(figsize=(8, 6))

    # Pure quadratic fit ax^2
    coeffs_quad = least_squares_poly(x_noisy, y_noisy, 2, only_x_squared=True)
    plot_regression(x_noisy, y_noisy, x_range, coeffs_quad, 'f1(x) = ax^2', only_x_squared=True)

    # Quadratic fit ax^2 + bx + c
    coeffs_quad_lin = least_squares_poly(x_noisy, y_noisy, 2)
    plot_regression(x_noisy, y_noisy, x_range, coeffs_quad_lin, 'f2(x) = ax^2 + bx + c')

    # 4th degree polynomial fit ax^4 + bx^3 + cx^2 + dx + e
    coeffs_degree4 = least_squares_poly(x_noisy, y_noisy, 4)
    plot_regression(x_noisy, y_noisy, x_range, coeffs_degree4, 'f3(x) = ax^4 + bx^3 + cx^2 + dx + e')

    plt.scatter(x_noisy, y_noisy, color='red', label='Noisy Data')
    plt.plot(x_range, func(x_range), label='True Function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Polynomial Regression Fit')
    plt.legend()
    plt.show()

# Perform regression on the interval [0, 5]
perform_and_plot(0, 5)

# Perform regression on the interval [5, 10]
perform_and_plot(5, 10)
