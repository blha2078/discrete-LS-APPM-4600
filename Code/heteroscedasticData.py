import numpy as np
import matplotlib.pyplot as plt

## Creates heteroscedastic data based on an x array, objective function and variance function
def generate_heteroscedastic_data(x, objective_func, variance_func):
    # Generate true y values based on the objective function
    y_true = objective_func(x)
    
    # Generate variance values using the variance function
    variance = variance_func(x)
    
    # Generate heteroscedastic noise
    noise = np.random.normal(0, np.sqrt(variance), len(x))
    
    # Add noise to y values to create observed values
    y_observed = y_true + noise
    
    return y_true, y_observed


# Define trial functions
f1 = lambda x: x
f2 = lambda x: np.sin(2 * np.pi * x)  # Sine function
f3 = lambda x: x**2 + 2*x + 1  # Quadratic function
f4 = lambda x: np.log(x + 1)  # Logarithmic function 
# Define function used
objective_func = f1

# Define your variance function
variance_func = lambda x: 0.5 * x

# num points
n = 500

# Generate x values
x = np.linspace(0, 1, n)

# Generate heteroscedastic data using the function
y_true, y_observed = generate_heteroscedastic_data(x, objective_func, variance_func)

# Plot the data
plt.figure(figsize=(8, 6))
plt.scatter(x, y_observed, label='Observed Data')
plt.plot(x, y_true, color='red', label='True Relationship')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Heteroscedastic Data')
plt.legend()
plt.grid(True)
plt.show()
