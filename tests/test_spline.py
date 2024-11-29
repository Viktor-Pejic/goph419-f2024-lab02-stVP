import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from src.linalg_interp import spline_function

xd = np.array(np.linspace(-10, 10, 10), dtype=float)  # x values
x = np.linspace(-10, 10, 100)  # Test x values

if __name__ == '__main__':
    #Test 1: Visualize spline accuracy
    y_linear = np.array(3*xd + 2)
    y_quad = np.array(xd**2 + 3*xd + 2)
    y_cubic = np.array(xd**3 - xd**2 + 3*xd + 2)


    f_linear = spline_function(xd, y_linear, order=1)
    lin_values = f_linear(x)

    f_quad = spline_function(xd, y_quad, order=2)
    quad_values = f_quad(x)

    f_cubic = spline_function(xd, y_cubic, order=3)
    cubic_values = f_cubic(x)


    fig, axs = plt.subplots(1, 3, figsize=(10, 5))

    axs[0].plot(x, lin_values, label = "Linear Spline")
    axs[0].scatter(xd, y_linear, color='red', label="Data Points")
    axs[0].title.set_text('Linear Spline')
    axs[0].legend()

    axs[1].plot(x, quad_values, label = "Quadratic Spline")
    axs[1].scatter(xd, y_quad, color='red', label="Data Points")
    axs[1].title.set_text('Quadratic Spline')
    axs[1].legend()

    axs[2].plot(x, cubic_values, label = "Cubic Spline")
    axs[2].scatter(xd, y_cubic, color='red', label="Data Points")
    axs[2].title.set_text('Cubic Spline')
    axs[2].legend()
    #plt.savefig('C:/Users/Viktor/repos/goph419-f2024-lab02-stVP/figures/Spline_Visualization.png')

if __name__ == '__main__':
    #Test 2: Compare against scipy.interpolate.UnivariateSpline using exponential function
    xd_2 = np.array(np.linspace(0, 10, 10), dtype=float)  # x values
    x_2 = np.linspace(0, 10, 100)  # Test x values
    y_exp = np.exp(xd_2) #exponential y values

    f_cubic_exp = spline_function(xd_2, y_exp, order=3)
    cubic_exp_values = f_cubic_exp(x_2)

    f_cubic_scipy_values = UnivariateSpline(xd_2, y_exp, k=3, ext='raise')
    f_cubic_scipy = f_cubic_scipy_values(x_2)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].plot(x_2, cubic_exp_values, label="Cubic Spline")
    axs[0].scatter(xd_2, y_exp, color='red', label="Data Points")
    axs[0].title.set_text('Iterated Cubic Spline')
    axs[0].legend()

    axs[1].plot(x_2, f_cubic_scipy, label="Scipy Generated Cubic Spline")
    axs[1].scatter(xd_2, y_exp, color='red', label="Data Points")
    axs[1].title.set_text('Scipy Generated Cubic Spline')
    axs[1].legend()

    plt.savefig('C:/Users/Viktor/repos/goph419-f2024-lab02-stVP/figures/Scipy_Spline_Comparison.png')
