import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x):
    return 0.25*x**2 + 3*np.sin(x) - 2*x + 5

def sin_series(x, a, b, c):
    return a*np.sin(x) + b*np.sin(3*x) + c*np.sin(5*x)

def least_squares_poly(x, y, degree):
    X = np.vander(x, degree + 1, increasing=True)
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    print(coeffs)
    return coeffs

def plot_fit_and_errors(x_noisy, y_noisy, x_range, y_true, models, noise_type, noise_level, interval):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))
    ax1.scatter(x_noisy, y_noisy, color='red', label='Noisy Data', s=5)
    ax1.plot(x_range, y_true, label='True Function', color='green', linestyle='--')

    # Fit models and plot
    for name, degree in models.items():
        if 'sin' in name:
            popt, _ = curve_fit(sin_series, x_noisy, y_noisy, p0=[5, 0.75, .25])
            y_fit = sin_series(x_range, *popt)
            print(popt)
        else:
            coeffs = least_squares_poly(x_noisy, y_noisy, degree)
            y_fit = np.polyval(coeffs[::-1], x_range)

        ax1.plot(x_range, y_fit, label=f'Fit: {name}')
        residuals_model = np.abs(y_fit - func(x_range))
        residuals_noise = np.abs(y_fit - y_noisy)

        ax2.plot(x_range, residuals_model, label=f'Model Error: {name}')
        ax3.plot(x_range, residuals_noise, label=f'Noise Error: {name}')

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

    '''
    ax3.set_title('Absolute Error to the Noisy Data')
    ax3.set_xlabel('x')
    ax3.set_ylabel('Error')
    ax3.legend(fontsize='small')
    ax3.grid(True)
    '''
    plt.show()

def perform_and_plot(interval, noise_type='normal', noise_level=3):
    x_noisy = np.linspace(interval[0], interval[1], 100)
    y_true = func(x_noisy)
    
    if noise_type == 'normal':
        noise = np.random.normal(loc=0, scale=noise_level, size=len(x_noisy))
    elif noise_type == 'uniform':
        noise = np.random.uniform(-noise_level, noise_level, size=len(x_noisy))
    y_noisy = y_true + noise

    models = {'ax^2 + bx + c': 2, 'ax^4 + bx^3 + cx^2 + dx + e': 4, 'sin(x) + sin(5x) + sin(9x)': 'sin_series'}
    plot_fit_and_errors(x_noisy, y_noisy, x_noisy, y_true, models, noise_type, noise_level, f'[{interval[0]}, {interval[1]}]')

# Usage
perform_and_plot([5, 10], 'normal', 1)