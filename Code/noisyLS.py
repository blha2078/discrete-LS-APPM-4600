import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x):
    return 0.25*x**2 + 3*np.sin(x) - 2*x + 5

def sin_series(x, a, b,c,d):
    return a*np.sin(x) + b*np.cos(x)+c*np.sin(0.5*x)+d*np.cos(0.5*x)

def least_squares_poly(x, y, degree):
    X = np.vander(x, degree + 1, increasing=True)
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    print(f"Degree {degree} coeffs:", coeffs )
    print('\n')
    return coeffs

def plot_fit_and_errors(x_noisy, y_noisy, x_range, y_true, models, noise_type, noise_level, interval):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
    ax1.scatter(x_noisy, y_noisy, color='red', label='Noisy Data', s=5)
    ax1.plot(x_range, y_true, label='True Function', color='green', linestyle='--')

    # Fit models and plot
    for name, degree in models.items():
        if 'sin' in name:
            popt, _ = curve_fit(sin_series, x_noisy, y_noisy, p0=[1,1,1,1])
            y_fit = sin_series(x_range, *popt)
            print("Sin coefficients:",popt)
            print('\n')
        else:
            coeffs = least_squares_poly(x_noisy, y_noisy, degree)
            y_fit = np.polyval(coeffs[::-1], x_range)

        ax1.plot(x_range, y_fit, label=f'Fit: {name}')
        residuals_model = np.abs(y_fit - func(x_range))
        residuals_noise = np.abs(y_fit - y_noisy)

        ax2.plot(x_range, residuals_model, label=f'Model Error: {name}')
        #ax3.plot(x_range, residuals_noise, label=f'Noise Error: {name}')

    # Set titles, labels, and legends
    ax1.set_title(f'Polynomial and Sine Series Regression: {noise_type.capitalize()} Noise (Level={noise_level}) on Interval {interval}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend(fontsize='small')
    ax1.grid(True)

    ax2.set_title('Absolute Error to the Model')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Error')
    ax2.legend(fontsize='small')
    ax2.grid(True)
    ax2.set_ylim(0, 8)

    '''
    ax3.set_title('Absolute Error to the Noisy Data')
    ax3.set_xlabel('x')
    ax3.set_ylabel('Error')
    ax3.legend(fontsize='small')
    ax3.grid(True)
    '''
    plt.show()

def perform_and_plot(interval, noise_type='normal', noise_level=3):
    x_noisy = np.linspace(interval[0], interval[1], 20*(interval[1]-interval[0]))
    y_true = func(x_noisy)
    
    if noise_type == 'normal':
        noise = np.random.normal(loc=0, scale=noise_level, size=len(x_noisy))
    elif noise_type == 'uniform':
        noise = np.random.uniform(-3*noise_level, 3*noise_level, size=len(x_noisy))
    y_noisy = y_true + noise

    models = {'ax^2 + bx + c': 2, 'ax^4 + bx^3 + cx^2 + dx + e': 4, 'asin(x) + bcos(x)+csin(0.5x)+dcos(0.5)x': 'sin_series'}
    plot_fit_and_errors(x_noisy, y_noisy, x_noisy, y_true, models, noise_type, noise_level, f'[{interval[0]}, {interval[1]}]')

# Usage
# print("Section 1")
# perform_and_plot([0, 5], 'normal', 1)
# perform_and_plot([5, 10], 'normal', 1)

# print("\nSection 2")
# perform_and_plot([0, 10], 'normal', 3)
# perform_and_plot([0, 10], 'uniform', 3)


print("\nSection 3")
perform_and_plot([0, 10], 'normal', 3)
perform_and_plot([0, 10], 'uniform', 3)
perform_and_plot([0, 10], 'normal', 1)
perform_and_plot([0, 10], 'normal', 5)