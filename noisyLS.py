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

def plot_regression(y_noisy,y_data, x_range, coeffs, label,ax1,ax2,ax3, only_x_squared=False):
    """ Plot regression curve along with the original data. """
    if only_x_squared:
        y_fit = coeffs[0]*x_range**2
    else:
        y_fit = np.polyval(coeffs, x_range)
        
    residuals_model = y_data - y_fit 
    residuals_noise = y_noisy - y_fit
    
    ax1.plot(x_range, y_fit, label=label, linestyle='--')
    ax1.grid(True)
    
    ax2.plot(x_range,np.abs(residuals_model),label = f'Error for {label}', linestyle = '-.')
    ax2.grid(True)
    
    ax3.plot(x_range,np.abs(residuals_noise),label = f'Error for {label}', linestyle = '-.')
    ax3.grid(True)

def perform_and_plot(x_start, x_end):
    x_noisy = np.linspace(x_start, x_end, 100)
    noise = np.random.normal(loc=0, scale=3, size=len(x_noisy))
    y_noisy = func(x_noisy) + noise
    y_true = func(x_noisy)
    x_range = np.linspace(x_start, x_end, 100)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

    # Pure quadratic fit ax^2
    coeffs_quad = least_squares_poly(x_noisy, y_noisy, 2, only_x_squared=True)
    plot_regression(y_noisy, y_true, x_range, coeffs_quad, 'f1(x) = ax^2', ax1, ax2, ax3, only_x_squared=True)

    # Quadratic fit ax^2 + bx + c
    coeffs_quad_lin = least_squares_poly(x_noisy, y_noisy, 2)
    plot_regression(y_noisy, y_true, x_range, coeffs_quad_lin, 'f2(x) = ax^2 + bx + c', ax1, ax2, ax3)

    # 4th degree polynomial fit ax^4 + bx^3 + cx^2 + dx + e
    coeffs_degree4 = least_squares_poly(x_noisy, y_noisy, 4)
    plot_regression(y_noisy, y_true, x_range, coeffs_degree4, 'f3(x) = ax^4 + bx^3 + cx^2 + dx + e', ax1, ax2, ax3)

    ax1.scatter(x_noisy, y_noisy, color='red', label='Noisy Data', s=5)
    ax1.plot(x_range, func(x_range), label='True Function')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Polynomial Regression Fit')
    ax1.legend()

    ax2.set_xlabel('x')
    ax2.set_ylabel('Error')
    ax2.set_title('Error for Model')
    ax2.legend()

    ax3.set_xlabel('x')
    ax3.set_ylabel('Error')
    ax3.set_title('Error for Noise')
    ax3.legend()

    ax1.set_xticks(np.arange(x_start, x_end + 1, 1))
    ax2.set_xticks(np.arange(x_start, x_end + 1, 1))
    ax3.set_xticks(np.arange(x_start, x_end + 1, 1))

    plt.show()

# Perform regression on the interval [0, 5]
perform_and_plot(0, 5)

# Perform regression on the interval [5, 10]
perform_and_plot(5, 10)
