import numpy as np
import matplotlib.pyplot as plt


def least_squares_poly(x, y, type):
    if type == "degree4":
        X = np.vander(x,5) 
    elif type == "quad":
        X = np.vstack((x**2, x, np.ones_like(x))).T
    elif type == "quadlin":
        X = np.vstack((x**2, np.ones_like(x))).T 
    else:
        return 0
    
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    
    return coeffs

# Define the true function y = x^2
def func(x):
    return x**2

# Generate noiseless data
x_noiseless = np.array([0,1,2,3,4,5,6,7,8,9,10])
true_y =  np.array([0,1,4,9,16,25,36,49,64,81,100])
y_outlier = np.array([0,1,4,9,16,85,36,49,64,81,100])



# Perform regression on noiseless data
degree4_coeff = least_squares_poly(x_noiseless, y_outlier, "degree4")
quad_coeff = least_squares_poly(x_noiseless, y_outlier, "quad")
quad_lin_coeff = least_squares_poly(x_noiseless, y_outlier, "quadlin")

# Plot the noiseless data, the outlier, and the regression lines
plt.figure(figsize=(8, 6))
plt.plot(x_noiseless, true_y, label='True Function')
plt.scatter(x_noiseless, y_outlier, color='red', label='Outlier')

# Plot quadratic-linear regression
quad_lin_y = quad_lin_coeff[0] * x_noiseless**2 
plt.plot(x_noiseless, quad_lin_y, label='f1(x) = ax^2', linestyle='--')
quad_lin_res = true_y - quad_lin_y

# Plot quadratic regression
quad_y = quad_coeff[0] * x_noiseless**2 + quad_coeff[1] * x_noiseless + quad_coeff[2]
plt.plot(x_noiseless, quad_y, label='f2(x) = ax^2 + bx + c', linestyle='-.')
quad_res = true_y - quad_y

# Plot degree 4 polynomial regression
degree4_y = sum(degree4_coeff[i] * x_noiseless**(4-i) for i in range(5))
plt.plot(x_noiseless, degree4_y, label='f3(x) = ax^5 + bx^4 + cx^3 + dx^2 + ex + f', linestyle=':')
deg4_res = true_y - degree4_y

plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression on Noiseless Data with Outlier')
plt.legend()
plt.grid(True)
plt.show()

#Plot Errors
plt.plot(x_noiseless, np.abs(quad_lin_res), label='f1(x) = ax^2', linestyle='--')
plt.plot(x_noiseless, np.abs(quad_res), label='f2(x) = ax^2 + bx + c', linestyle='-.')
plt.plot(x_noiseless, deg4_res, label='f3(x) = ax^5 + bx^4 + cx^3 + dx^2 + ex + f', linestyle=':')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression on Noiseless Data with Outlier')
plt.legend()
plt.grid(True)
plt.show()


def generate_data(x, function, outlier_index=None, outlier_value=None):
    y = function(x)
    if outlier_index is not None:
        y[outlier_index] = outlier_value
    return y

x_noiseless = np.linspace(0,10,21)
true_y = func(x_noiseless)
y_outlier = generate_data(x_noiseless,func,10, 85)
print(y_outlier)

# Perform regression on noiseless data
degree4_coeff = least_squares_poly(x_noiseless, y_outlier, "degree4")
quad_coeff = least_squares_poly(x_noiseless, y_outlier, "quad")
quad_lin_coeff = least_squares_poly(x_noiseless, y_outlier, "quadlin")

# Plot the noiseless data, the outlier, and the regression lines
plt.figure(figsize=(8, 6))
plt.plot(x_noiseless, true_y, label='True Function')
plt.scatter(x_noiseless, y_outlier, color='red', label='Outlier')

# Plot quadratic-linear regression
quad_lin_y = quad_lin_coeff[0] * x_noiseless**2 
plt.plot(x_noiseless, quad_lin_y, label='f1(x) = ax^2', linestyle='--')
quad_lin_res = true_y - quad_lin_y

# Plot quadratic regression
quad_y = quad_coeff[0] * x_noiseless**2 + quad_coeff[1] * x_noiseless + quad_coeff[2]
plt.plot(x_noiseless, quad_y, label='f2(x) = ax^2 + bx + c', linestyle='-.')
quad_res = true_y - quad_y

# Plot degree 4 polynomial regression
degree4_y = sum(degree4_coeff[i] * x_noiseless**(4-i) for i in range(5))
plt.plot(x_noiseless, degree4_y, label='f3(x) = ax^5 + bx^4 + cx^3 + dx^2 + ex + f', linestyle=':')
deg4_res = true_y - degree4_y

plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression on Noiseless Data with Outlier')
plt.legend()
plt.grid(True)
plt.show()

#Plot Errors
plt.plot(x_noiseless, np.abs(quad_lin_res), label='f1(x) = ax^2', linestyle='--')
plt.plot(x_noiseless, np.abs(quad_res), label='f2(x) = ax^2 + bx + c', linestyle='-.')
plt.plot(x_noiseless, deg4_res, label='f3(x) = ax^5 + bx^4 + cx^3 + dx^2 + ex + f', linestyle=':')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression on Noiseless Data with Outlier')
plt.legend()
plt.grid(True)
plt.show()
