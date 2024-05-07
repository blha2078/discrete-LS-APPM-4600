import numpy as np
import matplotlib.pyplot as plt

''' 
f : function we are analyzing
a : left bound for data points
b : right bound for data points
N : number of data points to generate
noise: whether errors for data points are generated from a 'normal' or 'uniform' distribution
sigma: standard deviation for error generation
points: whether x coordinates from points are equispaced across [a,b] or randomly from a uniform distribution on [a,b]
'''


def generate_data(f, a, b, N, noise='normal', sigma=1, points='equispaced'):
    if points == 'equispaced':
        x = np.linspace(a, b, N)
    elif points == 'random':
         x = np.random.uniform(a, b, N)
    else:
        print('Invalid Point Input')
        exit(0)

    if noise == 'normal':
        residuals = np.random.normal(0, sigma, N)
    elif noise == 'uniform':
        bounds = sigma * np.sqrt(3)
        residuals = np.random.uniform(-bounds, bounds, N)
    else:
        print('Invalid Error Input')
        exit(0)

    y = f(x) + residuals

    return np.array(list(zip(x, y)))


def least_squares(data):

    x_data = data[:, 0]
    y_data = data[:, 1]

    X = np.column_stack([x_data ** 2, np.sin(x_data), x_data, np.ones_like(x_data)])
    y = y_data

    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]

    return coeffs


def sensitivity_to_number_points():
    f = lambda x: 0.25 * (x ** 2) + (3 * np.sin(x)) + (-2 * x) + 5

    a = 0
    b = 10
    # N = [5, 10, 25, 50, 100, 250]

    N = [10, 50, 100]
    x = np.linspace(a, b, 100)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    ax1.plot(x, f(x), color='black', label='True Function')

    colors = ['blue', 'green', 'red']

    for i in range(len(N)):
        data = generate_data(f, a, b, N[i], sigma=2, points='equispaced')

        coeffs = least_squares(data)
        reg = lambda x: (coeffs[0] * x ** 2) + (coeffs[1] * np.sin(x)) + (coeffs[2] * x) + coeffs[3]

        print(f'Coefficients for {N[i]}: {coeffs}')

        ax1.plot(x, reg(x), color=colors[i], linestyle='dotted', label=f'Regression Line for {N[i]} Data Points')
        ax2.plot(x, np.abs((f(x) - reg(x)) / f(x)), color=colors[i], linestyle='dotted', label=f'Relative Error for {N[i]} Data Points')

    print('\n')

    ax1.legend()
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title('Effects of Varying Data Points on OLS Fit')
    ax1.grid(True)
    ax2.set_xlabel('x')
    ax2.set_ylabel('Relative Error')
    ax2.set_title('Effects of Varying Data Points on OLS Relative Error')
    ax2.grid(True)
    ax2.legend()
    plt.show()

    return


def senstivity_to_variance():
    f = lambda x: 0.25 * (x ** 2) + (3 * np.sin(x)) + (-2 * x) + 5

    a = 0
    b = 10

    Sigma = [5, 10, 25]
    x = np.linspace(a, b, 100)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    ax1.plot(x, f(x), color='black', label='True Function')

    colors = ['blue', 'green', 'red']

    residuals = np.random.normal(0, 1, 100)

    for i in range(len(Sigma)):
        data = np.array(list(zip(x, f(x) + residuals * Sigma[i])))

        coeffs = least_squares(data)
        reg = lambda x: (coeffs[0] * x ** 2) + (coeffs[1] * np.sin(x)) + (coeffs[2] * x) + coeffs[3]

        ax1.plot(x, reg(x), color=colors[i], linestyle='dotted', label=f'Regression Line for Variance = {Sigma[i]}')
        ax2.plot(x, np.abs((f(x) - reg(x)) / f(x)), color=colors[i], linestyle='dotted', label=f'Relative Error for Variance = {Sigma[i]} ')

    ax1.legend()
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title('Effects of Variance of Noise on OLS Fit')
    ax1.grid(True)
    ax2.set_xlabel('x')
    ax2.set_ylabel('Relative Error')
    ax2.set_title('Effects of Variance of Noise on OLS Relative Error')
    ax2.grid(True)
    ax2.legend()
    plt.show()

    return


if __name__ == '__main__':
    sensitivity_to_number_points()
    sensitivity_to_number_points()
    sensitivity_to_number_points()
    sensitivity_to_number_points()
    sensitivity_to_number_points()
    #senstivity_to_variance()
