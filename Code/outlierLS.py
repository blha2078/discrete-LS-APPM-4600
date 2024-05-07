import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def true_function(x):
    return 0.25*x**2 + 3*np.sin(x) - 2*x + 5

def sin_series(x, a, b, c, d):
    return a*np.sin(x) + b*np.cos(x) + c*np.sin(0.5*x) + d*np.cos(0.5*x)

# Function to generate data with optional outlier
def generate_data(x, function, outlier_index=None, outlier_value=None):
    y = function(x)
    if outlier_index is not None:
        y[outlier_index] = outlier_value
    return y

# Least squares polynomial regression
def least_squares_poly(x, y, degree):
    X = np.vander(x, degree + 1, increasing=True)
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    return coeffs

# Plot function and regression lines
def plot_regression(x, true_y, y_outlier, quad_lin_coeff, quad_coeff, degree4_coeff, sin_coeff, count):
    plt.figure(figsize=(12, 8))
    plt.plot(x, true_y, label='True Function')
    plt.scatter(x, y_outlier, color='red', label='Outlier')

    # Plot quadratic-linear regression
    quad_lin_y = np.polyval(quad_lin_coeff[::-1], x)
    #plt.plot(x, quad_lin_y, label='ax^2+bx+c', linestyle='--')

    # Plot quadratic regression
    quad_y = np.polyval(quad_coeff[::-1], x)
    plt.plot(x, quad_y,  label='ax^2+bx+c', linestyle='-.')

    # Plot degree 4 polynomial regression
    degree4_y = np.polyval(degree4_coeff[::-1], x)
    plt.plot(x, degree4_y, label='ax^4+bx^3+cx^2+dx+e', linestyle=':')

    # Plot sine function regression
    sin_y = sin_series(x, *sin_coeff)
    plt.plot(x, sin_y,label='asin(x)+bcos(x)+csin(0.5x)+dcos(0.5x)', linestyle='-.', color='purple')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Regression on Data with Outlier for {count} points')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot errors
def plot_errors(x, quad_lin_res, quad_res, deg4_res, sin_res, count):
    plt.plot(x, np.abs(quad_res), label='ax^2+bx+c', linestyle='-.')
    plt.plot(x, np.abs(deg4_res), label='ax^4+bx^3+cx^2+dx+e', linestyle=':')
    plt.plot(x, np.abs(sin_res), label='asin(x)+bcos(x)+csin(0.5x)+dcos(0.5x)', linestyle='-.', color='purple')
    
    plt.xlabel('x')
    plt.ylabel('Residuals')
    plt.title(f'Errors of Different Regressions for {count} points')
    plt.legend()
    plt.grid(True)
    plt.show()

# 10 data points
x_noiseless = np.linspace(0, 10, 11)
true_y = true_function(x_noiseless)
y_outlier = generate_data(x_noiseless, true_function, outlier_index=5, outlier_value=12)

# Perform least squares polynomial regression for different degrees
quad_lin_coeff = least_squares_poly(x_noiseless, y_outlier, 2)
quad_coeff = least_squares_poly(x_noiseless, y_outlier, 2)
degree4_coeff = least_squares_poly(x_noiseless, y_outlier, 4)

# Fit sine series using curve_fit
popt, _ = curve_fit(sin_series, x_noiseless, y_outlier, p0=[1, 1, 1, 1])

# Plot regression lines and errors
plot_regression(x_noiseless, true_y, y_outlier, quad_lin_coeff, quad_coeff, degree4_coeff, popt, 10)

# Calculate residuals
quad_lin_res = true_y - np.polyval(quad_lin_coeff[::-1], x_noiseless)
quad_res = true_y - np.polyval(quad_coeff[::-1], x_noiseless)
deg4_res = true_y - np.polyval(degree4_coeff[::-1], x_noiseless)
sin_res = true_y - sin_series(x_noiseless, *popt)

# Plot errors
plot_errors(x_noiseless, quad_lin_res, quad_res, deg4_res, sin_res, 10)


# 25 data points
x_noiseless = np.linspace(0, 10, 25)
true_y = true_function(x_noiseless)
y_outlier = generate_data(x_noiseless, true_function, outlier_index=12, outlier_value=12)

# Perform least squares polynomial regression for different degrees
quad_lin_coeff = least_squares_poly(x_noiseless, y_outlier, 2)
quad_coeff = least_squares_poly(x_noiseless, y_outlier, 2)
degree4_coeff = least_squares_poly(x_noiseless, y_outlier, 4)

# Fit sine series using curve_fit
popt, _ = curve_fit(sin_series, x_noiseless, y_outlier, p0=[1, 1, 1, 1])

# Plot regression lines and errors
plot_regression(x_noiseless, true_y, y_outlier, quad_lin_coeff, quad_coeff, degree4_coeff, popt, 25)

# Calculate residuals
quad_lin_res = true_y - np.polyval(quad_lin_coeff[::-1], x_noiseless)
quad_res = true_y - np.polyval(quad_coeff[::-1], x_noiseless)
deg4_res = true_y - np.polyval(degree4_coeff[::-1], x_noiseless)
sin_res = true_y - sin_series(x_noiseless, *popt)

# Plot errors
plot_errors(x_noiseless, quad_lin_res, quad_res, deg4_res, sin_res, 25)

# 50 data points
x_noiseless = np.linspace(0, 10, 51)
true_y = true_function(x_noiseless)
y_outlier = generate_data(x_noiseless, true_function, outlier_index=25, outlier_value=12)

# Perform least squares polynomial regression for different degrees
quad_lin_coeff = least_squares_poly(x_noiseless, y_outlier, 2)
quad_coeff = least_squares_poly(x_noiseless, y_outlier, 2)
degree4_coeff = least_squares_poly(x_noiseless, y_outlier, 4)

# Fit sine series using curve_fit
popt, _ = curve_fit(sin_series, x_noiseless, y_outlier, p0=[1, 1, 1, 1])

# Plot regression lines and errors
plot_regression(x_noiseless, true_y, y_outlier, quad_lin_coeff, quad_coeff, degree4_coeff, popt, 50)

# Calculate residuals
quad_lin_res = true_y - np.polyval(quad_lin_coeff[::-1], x_noiseless)
quad_res = true_y - np.polyval(quad_coeff[::-1], x_noiseless)
deg4_res = true_y - np.polyval(degree4_coeff[::-1], x_noiseless)
sin_res = true_y - sin_series(x_noiseless, *popt)

# Plot errors
plot_errors(x_noiseless, quad_lin_res, quad_res, deg4_res, sin_res, 50)

