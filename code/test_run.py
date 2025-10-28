# Import necessary libraries/modules
import xarray as xr
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from multiprocessing import Pool
import os

# Function to generate and save plot for a single timestep


def plot_timestep(args):
    timestep, data_file, times_str = args
    # Open dataset within the function to avoid sharing file handles across processes
    dataset = nc.Dataset(data_file)

    # Extract data for the given timestep
    u10 = dataset.variables['U10'][timestep, :, :]  # 10-m u-wind
    v10 = dataset.variables['V10'][timestep, :, :]  # 10-m v-wind
    xlat = dataset.variables['XLAT'][0, :, :]
    xlong = dataset.variables['XLONG'][0, :, :]

    # Calculate wind speed
    wind_speed = np.sqrt(u10**2 + v10**2)

    # Create plot
    fig, ax = plt.subplots(figsize=(20, 12), subplot_kw={
                           'projection': ccrs.LambertConformal()})
    ax.set_extent([xlong.min(), xlong.max(), xlat.min(),
                  xlat.max()], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, facecolor='white', edgecolor='black')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

    # Plot wind speed
    contour = ax.contourf(xlong, xlat, wind_speed, transform=ccrs.PlateCarree(),
                          cmap='berlin', levels=np.linspace(0, 60, 21))
    plt.colorbar(contour, label='Wind Speed (m/s)')
    plt.title(f'10-m Wind Speed at {times_str[timestep]}')

    # Save plot
    output_file = f'wind_speed_plot_timestep_{timestep:03d}.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close figure to free memory

    # Close dataset
    dataset.close()
    print(f"Saved plot for timestep {timestep}: {output_file}")
    return output_file

# Main script


def main():
    # Load the netCDF data file
    data = 'data/wrfout_d03_2022-09-26_Ian2022_UNCPL.nc'
    dataset = nc.Dataset(data)

    # Extract time information
    times = dataset.variables['Times'][:]
    times_str = [''.join(time.astype(str)) if not np.ma.is_masked(
        time) else '' for time in times]
    num_timesteps = len(times)
    print(f"Number of timesteps: {num_timesteps}")
    print(f"First timestamp: {times_str[0]}")
    print(f"Last timestamp: {times_str[-1]}")

    # Close dataset to avoid file handle issues
    dataset.close()

    # Prepare arguments for parallel processing
    tasks = [(i, data, times_str) for i in range(num_timesteps)]

    # Use multiprocessing Pool with 8 workers
    with Pool(processes=8) as pool:
        results = pool.map(plot_timestep, tasks)

    print("All plots generated:", results)


if __name__ == '__main__':
    main()
