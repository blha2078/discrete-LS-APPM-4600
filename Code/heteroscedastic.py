import numpy as np
import matplotlib.pyplot as plt

# num points
n = 500

# [0, 1]
x = np.linspace(0, 1, n)

# Define trial functions
f1 = lambda x: x
f2 = lambda x: np.sin(2 * np.pi * x)  # Sine function
f3 = lambda x: x**2 + 2*x + 1  # Quadratic function
f4 = lambda x: np.log(x + 1)  # Logarithmic function 


# Generate y values (true values) based on the lambda function
objective_func = f2
y_true = objective_func(x)

# Generate heteroscedastic noise
# higher constant means more noise
variance = 0.5 * x

# Generate random noise with increasing variance
noise = np.random.normal(0, np.sqrt(variance), n)

# Add noise to y values to create the observed values
y_observed = y_true + noise

# Plot the data
plt.figure(figsize=(8, 6))
plt.scatter(x, y_observed, label='Observed Data')
plt.plot(x, y_true, color='red', label='True Relationship: y = x')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Heteroscedastic Data')
plt.legend()
plt.grid(True)
plt.show()
