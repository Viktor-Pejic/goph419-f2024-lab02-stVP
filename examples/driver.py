import numpy as np
from matplotlib import pyplot as plt


from src.linalg_interp import spline_function
def main():
    air_density_data = np.loadtxt("C:/Users/Viktor/repos/goph419-f2024-lab02-stVP/data/air_density_vs_temp_eng_toolbox.txt", 'float')
    water_density_data = np.loadtxt("C:/Users/Viktor/repos/goph419-f2024-lab02-stVP/data/water_density_vs_temp_usgs.txt", 'float')

    x_values_air = air_density_data[:,0]
    y_values_air = air_density_data[:,1]

    x_values_water = water_density_data[:,0]
    y_values_water = water_density_data[:,1]

    x_plot_air = np.linspace(min(x_values_air), max(x_values_air), 100)

    x_plot_water = np.linspace(min(x_values_water), max(x_values_water), 100)

    linear_air = spline_function(x_values_air, y_values_air, order=1)
    f_linear_air = linear_air(x_plot_air)

    linear_water = spline_function(x_values_water, y_values_water, order=1)
    f_linear_water = linear_water(x_plot_water)

    fig, axs = plt.subplots(3, 2, figsize=(10,12))
    plt.subplots_adjust(hspace=0.3, wspace=0.5)
    plt.suptitle("Air Density vs Temperature and Water Density vs Temperature", fontsize=15, fontweight='bold')

    axs[0,0].plot(x_plot_air, f_linear_air, 'b', label='Linear air density vs temp')
    axs[0,0].scatter(x_values_air, y_values_air, color='red', label='Data points')
    axs[0,0].set_title('Linear', fontweight='bold')
    axs[0,0].set_xlabel('Temperature (C)')
    axs[0,0].set_ylabel('Air Density (kg/m^3)')

    axs[0,1].plot(x_plot_water, f_linear_water, 'g', label='Linear water density vs temp')
    axs[0,1].scatter(x_values_water, y_values_water, color='red', label='Data points')
    axs[0,1].set_title('Linear', fontweight='bold')
    axs[0,1].set_xlabel('Temperature (C)')
    axs[0,1].set_ylabel('Water Density (kg/m^3)')

    quadratic_air = spline_function(x_values_air, y_values_air, order=2)
    f_quad_air = quadratic_air(x_plot_air)

    quadratic_water = spline_function(x_values_water, y_values_water, order=2)
    f_quad_water = quadratic_water(x_plot_water)

    axs[1,0].plot(x_plot_air, f_quad_air,  'b', label='Quadratic air density vs temp')
    axs[1,0].scatter(x_values_air, y_values_air, color='red', label='Data points')
    axs[1,0].set_title('Quadratic', fontweight='bold')
    axs[1,0].set_xlabel('Temperature (C)')
    axs[1,0].set_ylabel('Air Density (kg/m^3)')

    axs[1,1].plot(x_plot_water, f_quad_water,  'g', label='Quadratic water density vs temp')
    axs[1,1].scatter(x_values_water, y_values_water, color='red', label='Data points')
    axs[1,1].set_title('Quadratic', fontweight='bold')
    axs[1,1].set_xlabel('Temperature (C)')
    axs[1,1].set_ylabel('Water Density (kg/m^3)')

    cubic_air = spline_function(x_values_air, y_values_air, order=3)
    f_cubic_air = cubic_air(x_plot_air)

    cubic_water = spline_function(x_values_water, y_values_water, order=3)
    f_cubic_water = cubic_water(x_plot_water)

    axs[2,0].plot(x_plot_air, f_cubic_air,  'b', label='Cubic air density vs temp')
    axs[2,0].scatter(x_values_air, y_values_air, color='red', label='Data points')
    axs[2,0].set_title('Cubic', fontweight='bold')
    axs[2,0].set_xlabel('Temperature (C)')
    axs[2,0].set_ylabel('Air Density (kg/m^3)')

    axs[2,1].plot(x_plot_water, f_cubic_water,  'g', label='Cubic water density vs temp')
    axs[2,1].scatter(x_values_water, y_values_water, color='red', label='Data points')
    axs[2,1].set_title('Cubic', fontweight='bold')
    axs[2,1].set_xlabel('Temperature (C)')
    axs[2,1].set_ylabel('Water Density (kg/m^3)')

    plt.savefig('C:/Users/Viktor/repos/goph419-f2024-lab02-stVP/figures/Air_Density_and_Water_Density_vs_Temperature.png')

if __name__ == '__main__':
    main()





