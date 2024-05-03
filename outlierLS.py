import numpy as np
import matplotlib.pyplot as plt

# Function to perform polynomial regression
def least_squares_poly(x, y, type):
    if type == "degree4":
        X = np.vander(x, 5)
    elif type == "quad":
        X = np.vstack((x**2, x, np.ones_like(x))).T
    elif type == "quadlin":
        X = np.vstack((x**2, np.ones_like(x))).T 
    elif type == "sin":
        X = np.sin(np.array([x, 3*x, 5*x, 7*x, 9*x])).T
    else:
        return None
    
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    return coeffs

# Define the true function y = x^2
def true_function(x):
    return x**2

def true_function2(x):
    return np.sin(x)

# Function to generate data with optional outlier
def generate_data(x, function, outlier_index=None, outlier_value=None):
    y = function(x)
    if outlier_index is not None:
        y[outlier_index] = outlier_value
    return y

# Plot function and regression lines
def plot_regression(x, true_y, y_outlier, quad_lin_coeff, quad_coeff, degree4_coeff, sin_coeff, count):
    plt.figure(figsize=(8, 6))
    plt.plot(x, true_y, label='True Function')
    plt.scatter(x, y_outlier, color='red', label='Outlier')

    # Plot quadratic-linear regression
    quad_lin_y = quad_lin_coeff[0] * x**2 
    plt.plot(x, quad_lin_y, label='f1(x) = ax^2', linestyle='--')

    # Plot quadratic regression
    quad_y = quad_coeff[0] * x**2 + quad_coeff[1] * x + quad_coeff[2]
    plt.plot(x, quad_y, label='f2(x) = ax^2 + bx + c', linestyle='-.')

    # Plot degree 4 polynomial regression
    degree4_y = sum(degree4_coeff[i] * x**(4-i) for i in range(5))
    plt.plot(x, degree4_y, label='f3(x) = ax^4 + bx^3 + cx^2 + dx + e', linestyle=':')

    # Plot sine function regression
    sin_y = sin_coeff[0] * np.sin(x) + sin_coeff[1] * np.sin(3*x) + sin_coeff[2] * np.sin(5*x) + sin_coeff[3] * np.sin(7*x) + sin_coeff[4] * np.sin(9*x)
    plt.plot(x, sin_y, label='f4(x) = sin(x) + sin(3x) + sin(5x) + sin(7x) + sin(9x)', linestyle='-.', color='purple')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Regression on Data with Outlier for {count} points')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot errors
def plot_errors(x, quad_lin_res, quad_res, deg4_res, sin_res, count):
    plt.plot(x, np.abs(quad_lin_res), label='f1(x) = ax^2', linestyle='--')
    plt.plot(x, np.abs(quad_res), label='f2(x) = ax^2 + bx + c', linestyle='-.')
    plt.plot(x, np.abs(deg4_res), label='f3(x) = ax^4 + bx^3 + cx^2 + dx + e', linestyle=':')
    plt.plot(x, np.abs(sin_res), label='f4(x) = sin(x) + sin(3x) + sin(5x) + sin(7x) + sin(9x)', linestyle='-.', color='purple')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Errors in Regression for {count} points')
    plt.legend()
    plt.grid(True)
    plt.show()

# 10 data points
x_noiseless = np.linspace(0, 10, 11)
true_y = true_function(x_noiseless)
y_outlier = generate_data(x_noiseless, true_function, outlier_index=5, outlier_value=85)

degree4_coeff = least_squares_poly(x_noiseless, y_outlier, "degree4")
quad_coeff = least_squares_poly(x_noiseless, y_outlier, "quad")
quad_lin_coeff = least_squares_poly(x_noiseless, y_outlier, "quadlin")
sin_coeff = least_squares_poly(x_noiseless, y_outlier, "sin")

plot_regression(x_noiseless, true_y, y_outlier, quad_lin_coeff, quad_coeff, degree4_coeff, sin_coeff, 10)

quad_lin_res = true_y - (quad_lin_coeff[0] * x_noiseless**2)
quad_res = true_y - (quad_coeff[0] * x_noiseless**2 + quad_coeff[1] * x_noiseless + quad_coeff[2])
deg4_res = true_y - sum(degree4_coeff[i] * x_noiseless**(4-i) for i in range(5))
sin_res = true_y - (sin_coeff[0] * np.sin(x_noiseless) + sin_coeff[1] * np.sin(3*x_noiseless) + sin_coeff[2] * np.sin(5*x_noiseless) + sin_coeff[3] * np.sin(7*x_noiseless) + sin_coeff[4] * np.sin(9*x_noiseless))

plot_errors(x_noiseless, quad_lin_res, quad_res, deg4_res, sin_res, 10)


# 20 data points
x_noiseless = np.linspace(0, 10, 21)
true_y = true_function(x_noiseless)
y_outlier = generate_data(x_noiseless, true_function, outlier_index=10, outlier_value=85)

degree4_coeff = least_squares_poly(x_noiseless, y_outlier, "degree4")
quad_coeff = least_squares_poly(x_noiseless, y_outlier, "quad")
quad_lin_coeff = least_squares_poly(x_noiseless, y_outlier, "quadlin")
sin_coeff = least_squares_poly(x_noiseless, y_outlier, "sin")

plot_regression(x_noiseless, true_y, y_outlier, quad_lin_coeff, quad_coeff, degree4_coeff, sin_coeff, 20)

quad_lin_res = true_y - (quad_lin_coeff[0] * x_noiseless**2)
quad_res = true_y - (quad_coeff[0] * x_noiseless**2 + quad_coeff[1] * x_noiseless + quad_coeff[2])
deg4_res = true_y - sum(degree4_coeff[i] * x_noiseless**(4-i) for i in range(5))
sin_res = true_y - (sin_coeff[0] * np.sin(x_noiseless) + sin_coeff[1] * np.sin(3*x_noiseless) + sin_coeff[2] * np.sin(5*x_noiseless) + sin_coeff[3] * np.sin(7*x_noiseless) + sin_coeff[4] * np.sin(9*x_noiseless))

plot_errors(x_noiseless, quad_lin_res, quad_res, deg4_res, sin_res, 20)


#50 data points 

x_noiseless = np.linspace(0, 10, 51)
true_y = true_function(x_noiseless)
y_outlier = generate_data(x_noiseless, true_function, outlier_index=25, outlier_value=85)

degree4_coeff = least_squares_poly(x_noiseless, y_outlier, "degree4")
quad_coeff = least_squares_poly(x_noiseless, y_outlier, "quad")
quad_lin_coeff = least_squares_poly(x_noiseless, y_outlier, "quadlin")
sin_coeff = least_squares_poly(x_noiseless, y_outlier, "sin")

plot_regression(x_noiseless, true_y, y_outlier, quad_lin_coeff, quad_coeff, degree4_coeff, sin_coeff, 50)

quad_lin_res = true_y - (quad_lin_coeff[0] * x_noiseless**2)
quad_res = true_y - (quad_coeff[0] * x_noiseless**2 + quad_coeff[1] * x_noiseless + quad_coeff[2])
deg4_res = true_y - sum(degree4_coeff[i] * x_noiseless**(4-i) for i in range(5))
sin_res = true_y - (sin_coeff[0] * np.sin(x_noiseless) + sin_coeff[1] * np.sin(3*x_noiseless) + sin_coeff[2] * np.sin(5*x_noiseless) + sin_coeff[3] * np.sin(7*x_noiseless) + sin_coeff[4] * np.sin(9*x_noiseless))

plot_errors(x_noiseless, quad_lin_res, quad_res, deg4_res, sin_res, 50)

#Sin Implementation

# 10 data points
x_noiseless = np.linspace(0, 10, 11)
true_y = true_function2(x_noiseless)
y_outlier = generate_data(x_noiseless, true_function2, outlier_index=5, outlier_value=2)

degree4_coeff = least_squares_poly(x_noiseless, y_outlier, "degree4")
quad_coeff = least_squares_poly(x_noiseless, y_outlier, "quad")
quad_lin_coeff = least_squares_poly(x_noiseless, y_outlier, "quadlin")
sin_coeff = least_squares_poly(x_noiseless, y_outlier, "sin")

plot_regression(x_noiseless, true_y, y_outlier, quad_lin_coeff, quad_coeff, degree4_coeff, sin_coeff, 10)

quad_lin_res = true_y - (quad_lin_coeff[0] * x_noiseless**2)
quad_res = true_y - (quad_coeff[0] * x_noiseless**2 + quad_coeff[1] * x_noiseless + quad_coeff[2])
deg4_res = true_y - sum(degree4_coeff[i] * x_noiseless**(4-i) for i in range(5))
sin_res = true_y - (sin_coeff[0] * np.sin(x_noiseless) + sin_coeff[1] * np.sin(3*x_noiseless) + sin_coeff[2] * np.sin(5*x_noiseless) + sin_coeff[3] * np.sin(7*x_noiseless) + sin_coeff[4] * np.sin(9*x_noiseless))

plot_errors(x_noiseless, quad_lin_res, quad_res, deg4_res, sin_res, 10)


# 20 data points
x_noiseless = np.linspace(0, 10, 21)
true_y = true_function2(x_noiseless)
y_outlier = generate_data(x_noiseless, true_function2, outlier_index=10, outlier_value=2)

degree4_coeff = least_squares_poly(x_noiseless, y_outlier, "degree4")
quad_coeff = least_squares_poly(x_noiseless, y_outlier, "quad")
quad_lin_coeff = least_squares_poly(x_noiseless, y_outlier, "quadlin")
sin_coeff = least_squares_poly(x_noiseless, y_outlier, "sin")

plot_regression(x_noiseless, true_y, y_outlier, quad_lin_coeff, quad_coeff, degree4_coeff, sin_coeff, 20)

quad_lin_res = true_y - (quad_lin_coeff[0] * x_noiseless**2)
quad_res = true_y - (quad_coeff[0] * x_noiseless**2 + quad_coeff[1] * x_noiseless + quad_coeff[2])
deg4_res = true_y - sum(degree4_coeff[i] * x_noiseless**(4-i) for i in range(5))
sin_res = true_y - (sin_coeff[0] * np.sin(x_noiseless) + sin_coeff[1] * np.sin(3*x_noiseless) + sin_coeff[2] * np.sin(5*x_noiseless) + sin_coeff[3] * np.sin(7*x_noiseless) + sin_coeff[4] * np.sin(9*x_noiseless))

plot_errors(x_noiseless, quad_lin_res, quad_res, deg4_res, sin_res, 20)


#50 data points 

x_noiseless = np.linspace(0, 10, 51)
true_y = true_function2(x_noiseless)
y_outlier = generate_data(x_noiseless, true_function2, outlier_index=25, outlier_value=2)

degree4_coeff = least_squares_poly(x_noiseless, y_outlier, "degree4")
quad_coeff = least_squares_poly(x_noiseless, y_outlier, "quad")
quad_lin_coeff = least_squares_poly(x_noiseless, y_outlier, "quadlin")
sin_coeff = least_squares_poly(x_noiseless, y_outlier, "sin")

plot_regression(x_noiseless, true_y, y_outlier, quad_lin_coeff, quad_coeff, degree4_coeff, sin_coeff, 50)

quad_lin_res = true_y - (quad_lin_coeff[0] * x_noiseless**2)
quad_res = true_y - (quad_coeff[0] * x_noiseless**2 + quad_coeff[1] * x_noiseless + quad_coeff[2])
deg4_res = true_y - sum(degree4_coeff[i] * x_noiseless**(4-i) for i in range(5))
sin_res = true_y - (sin_coeff[0] * np.sin(x_noiseless) + sin_coeff[1] * np.sin(3*x_noiseless) + sin_coeff[2] * np.sin(5*x_noiseless) + sin_coeff[3] * np.sin(7*x_noiseless) + sin_coeff[4] * np.sin(9*x_noiseless))

plot_errors(x_noiseless, quad_lin_res, quad_res, deg4_res, sin_res, 50)