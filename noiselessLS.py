import numpy as np
import matplotlib.pyplot as plt 

x_range = np.linspace(0,5,100)

def func(x):
    return x**2

y = func(x_range)

def least_squares_poly(x,y,type):
    if type == "degree5":
        X = np.vander(x,5)
    elif type == "quad":
        X = np.vstack((x**2,x,np.ones_like(x))).T
    elif type == "quadlin":
        X = np.vstack((x**2,np.ones_like(x))).T 
    else:
        return 0
    
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    
    return coeffs

degree5_coeff = least_squares_poly(x_range,y,"degree5")
quad_coeff = least_squares_poly(x_range,y,"quad")
quad_lin_coeff= least_squares_poly(x_range,y,"quadlin")

print("Coefficient (a) for f1(x) = ax^2 : ",quad_lin_coeff[0])
print("Coefficients (a,b,c) for f2(x) = ax^2 + bx + c : ",quad_coeff)
print("Coefficeints (a, b, c, d, e) for f3(x) = ax^4 + bx^3 + cx^2 + dx + e: ", degree5_coeff)


plt.figure(figsize=(8, 6))

# Plot data points
plt.plot(x_range,y,label = 'True Function')

# Plot quadratic-linear regression
quad_lin_y = quad_lin_coeff[0] * x_range**2 
plt.plot(x_range, quad_lin_y, label='f1(x) = ax^2', linestyle='--')

# Plot quadratic regression
quad_y = quad_coeff[0] * x_range**2 + quad_coeff[1] * x_range + quad_coeff[2]
plt.plot(x_range, quad_y, label='f2(x) = ax^2 + bx + c', linestyle='-.')

# Plot degree 5 polynomial regression
degree5_y = sum(degree5_coeff[i] * x_range**(5-i) for i in range(5))
plt.plot(x_range, degree5_y, label='f3(x) = ax^4 + bx^3 + cx^2 + dx + e', linestyle=':')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Least Squares Regression')
plt.legend()
plt.grid(True)
plt.show()

#Repeat on interval [5,10]
x_range = np.linspace(5,10,100)
y = func(x_range)

plt.figure(figsize=(8, 6))

# Plot data points
plt.plot(x_range,y,label = 'True Function')

# Plot quadratic-linear regression
quad_lin_y = quad_lin_coeff[0] * x_range**2 
plt.plot(x_range, quad_lin_y, label='f1(x) = ax^2', linestyle='--')

# Plot quadratic regression
quad_y = quad_coeff[0] * x_range**2 + quad_coeff[1] * x_range + quad_coeff[2]
plt.plot(x_range, quad_y, label='f2(x) = ax^2 + bx + c', linestyle='-.')

# Plot degree 5 polynomial regression
degree5_y = sum(degree5_coeff[i] * x_range**(4-i) for i in range(4))
#plt.plot(x_range, degree5_y, label='f3(x) = ax^4 + bx^3 + cx^2 + dx + e', linestyle=':')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Least Squares Regression')
plt.legend()
plt.grid(True)
plt.show()
