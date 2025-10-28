# ---------------------------------------------------------------------------------------
# 1. Import Python libraries
# ---------------------------------------------------------------------------------------
import numpy as np
import netCDF4 as nc
from pathlib import Path
from pyproj import Proj, Transformer
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import binned_statistic
from datetime import datetime
import time

# ---------------------------------------------------------------------------------------
# 2. Load WRF data
# ---------------------------------------------------------------------------------------
t_start = time.time()
print("Time started at:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

print("Loading WRF data...")
data_path = Path('../data/wrfout_d03_2022-09-26_Ian2022_UNCPL.nc')
ds = nc.Dataset(data_path)
t = 60  # choose desired timestep

# Get lat/lon
lat = ds.variables['XLAT'][0, :, :]
lon = ds.variables['XLONG'][0, :, :]

# ---------------------------------------------------------------------------------------
# 3. Unstagger U10 and V10 components
# ---------------------------------------------------------------------------------------
# Check if U10/V10 are staggered
# print("U10 dims:", ds.variables['U10'].dimensions)
# print("V10 dims:", ds.variables['V10'].dimensions)

# Function: Unstagger WRF variables


def unstagger(var, axis):
    return 0.5 * (var.take(indices=range(var.shape[axis]-1), axis=axis) +
                  var.take(indices=range(1, var.shape[axis]), axis=axis))


u10 = ds.variables['U10'][t, :, :]
v10 = ds.variables['V10'][t, :, :]

if 'west_east_stag' in ds.variables['U10'].dimensions:
    u10 = unstagger(u10, axis=1)
if 'south_north_stag' in ds.variables['V10'].dimensions:
    v10 = unstagger(v10, axis=0)

u10 = np.array(u10)
v10 = np.array(v10)

# ---------------------------------------------------------------------------------------
# 4. Set initial best center to be minimum surface pressure (unchanged)
# ---------------------------------------------------------------------------------------
psfc = ds.variables['PSFC'][t, :, :]
imin_psfc = np.unravel_index(np.argmin(psfc), psfc.shape)
center_lat = float(lat[imin_psfc])
center_lon = float(lon[imin_psfc])
print(
    f"Initial center (min surface pressure): {center_lat:.2f}°, {center_lon:.2f}°")

# ---------------------------------------------------------------------------------------
# 5. Refine center based on eyewall (max tangential wind in RMW annulus)
# ---------------------------------------------------------------------------------------
# Define search parameters
search_radius_km = 200  # max radius for binning (focus on inner core)
annulus_width_km = 15  # half-width of annulus around RMW
search_box = 20  # grid points to search around initial center (+/- value)

# Get indices around initial center
i_min, j_min = imin_psfc
i_range = range(max(0, i_min - search_box),
                min(lat.shape[0], i_min + search_box + 1))
j_range = range(max(0, j_min - search_box),
                min(lat.shape[1], j_min + search_box + 1))

# Initialize variables to track max annulus mean tangential wind
max_annulus_mean = -np.inf
refined_lat = center_lat
refined_lon = center_lon

for i in i_range:  # Loop over candidate center latitudes
    for j in j_range:  # Loop over candidate center longitudes
        cand_lat = lat[i, j]
        cand_lon = lon[i, j]

        # Project to AEQD centered at candidate
        aeqd = Proj(proj='aeqd', lat_0=cand_lat, lon_0=cand_lon, datum='WGS84')
        transformer = Transformer.from_proj('epsg:4326', aeqd, always_xy=True)
        x, y = transformer.transform(lon, lat)

        # Compute radial distance and azimuthal angle
        r = np.sqrt(x**2 + y**2) / 1000.0  # km
        theta = np.arctan2(y, x)

        # Mask beyond search radius to focus on core
        mask = r <= search_radius_km
        r_masked = np.where(mask, r, np.nan)
        theta_masked = np.where(mask, theta, np.nan)
        u10_masked = np.where(mask, u10, np.nan)
        v10_masked = np.where(mask, v10, np.nan)

        # Tangential wind (positive for cyclonic in NH, negative in SH)
        tangential = -u10_masked * \
            np.sin(theta_masked) + v10_masked * np.cos(theta_masked)

        # Bin azimuthal mean tangential wind by radius
        # adjust bin count as needed
        bins = np.linspace(0, search_radius_km, 100)
        mean_tang, bin_edges, _ = binned_statistic(
            r_masked.flatten(), tangential.flatten(), statistic='mean', bins=bins)
        r_mid = (bin_edges[:-1] + bin_edges[1:]) / 2  # midpoints of bins

        # Find RMW and discard NaN values
        valid = ~np.isnan(mean_tang)
        if not np.any(valid):
            continue
        rmw = r_mid[valid][np.argmax(mean_tang[valid])]

        # Compute annulus around RMW
        annulus_mask = (r_masked >= rmw -
                        annulus_width_km) & (r_masked < rmw + annulus_width_km)
        if np.sum(annulus_mask) == 0:
            continue
        annulus_mean = np.nanmean(tangential[annulus_mask])

        # Update if this is the max
        if annulus_mean > max_annulus_mean:
            max_annulus_mean = annulus_mean
            refined_lat = cand_lat
            refined_lon = cand_lon

print(
    f"Refined center (eyewall-based max tangential wind): {refined_lat:.2f}°, {refined_lon:.2f}°")

# ---------------------------------------------------------------------------------------
# 6. Compute radial and tangential wind relative to refined center
# ---------------------------------------------------------------------------------------
# Project to AEQD centered at refined center
aeqd_refined = Proj(proj='aeqd', lat_0=refined_lat,
                    lon_0=refined_lon, datum='WGS84')
transformer_refined = Transformer.from_proj(
    'epsg:4326', aeqd_refined, always_xy=True)

# Compute x, y in meters relative to refined center
x_ref, y_ref = transformer_refined.transform(lon, lat)
r = np.sqrt(x_ref**2 + y_ref**2)
theta = np.arctan2(y_ref, x_ref)

# Compute radial and tangential wind components
radial_wind = u10 * np.cos(theta) + v10 * np.sin(theta)
tangential_wind = -u10 * np.sin(theta) + v10 * np.cos(theta)

# ---------------------------------------------------------------------------------------
# 7. Start plotting with Cartopy and Matplotlib
# ---------------------------------------------------------------------------------------
# Extract timestamp for timestep t
time_bytes = ds.variables['Times'][t]
time_str = ''.join([byte.decode('utf-8') for byte in time_bytes])
time_dt = datetime.strptime(time_str, '%Y-%m-%d_%H:%M:%S')
time_formatted = time_dt.strftime(
    '%H:%Mz, %b %d, %Y')  # e.g., '00:00z, Sep 26, 2022'

# Compute max wind and min pressure
wind_speed = np.sqrt(u10**2 + v10**2)
max_wind = np.max(wind_speed)
min_pressure = np.min(psfc) / 100  # Convert to hPa

# Create figure
fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={
                       'projection': ccrs.PlateCarree()})
ax.set_extent([lon.min(), lon.max(), lat.min(),
              lat.max()], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')

# Contour plot for wind speed
im = ax.contourf(lon, lat, wind_speed, levels=np.linspace(
    0, 60, 21), cmap='bone', transform=ccrs.PlateCarree())

# Quiver plot for wind vectors
ax.quiver(lon[::20, ::20], lat[::20, ::20], u10[::20, ::20],
          v10[::20, ::20], scale=1500, transform=ccrs.PlateCarree())

# Add radial circles around refined center (i.e., radii and colors)
radius_25_miles = 25 / 69
radius_50_miles = 50 / 69
radius_100_miles = 100 / 69
radius_150_miles = 150 / 69
radius_200_miles = 200 / 69

# Add radial circles
rad_25 = plt.Circle((refined_lon, refined_lat), radius_25_miles,
                    color="#ff32f5", fill=False, linewidth=2, label='25 miles')
rad_50 = plt.Circle((refined_lon, refined_lat), radius_50_miles,
                    color="#c00000", fill=False, linewidth=2, label='50 miles')
rad_100 = plt.Circle((refined_lon, refined_lat), radius_100_miles,
                     color="#ec4b00", fill=False, linewidth=2, label='100 miles')
rad_150 = plt.Circle((refined_lon, refined_lat), radius_150_miles,
                     color="#EED600", fill=False, linewidth=2, label='150 miles')
rad_200 = plt.Circle((refined_lon, refined_lat), radius_200_miles,
                     color="#12D400", fill=False, linewidth=2, label='200 miles')

ax.add_patch(rad_25)
ax.add_patch(rad_50)
ax.add_patch(rad_100)
ax.add_patch(rad_150)
ax.add_patch(rad_200)

# Add center marker
refined_center_dot, = ax.plot(refined_lon, refined_lat, 'o',
                              color="#A70075", markersize=10, label='Eyewall-Refined Center')

# Create legend
plt.legend(handles=[rad_25, rad_50, rad_100,
           rad_150, rad_200], loc='upper right')

# Add colorbar
cbar = fig.colorbar(im, ax=ax, orientation='vertical',
                    pad=0.02, aspect=20, shrink=0.7)
cbar.set_label('Wind Speed (m/s)', fontsize=10)

# Add titles
fig.text(0.125, 0.91, 'Hurricane IAN', ha='left', va='top', fontsize=12)
fig.text(0.77, 0.91, f'Lat: {refined_lat:.2f} | Lon: {refined_lon:.2f} | Highest Winds: {max_wind:.1f} m/s | Lowest Pressure: {min_pressure:.1f} hPa',
         ha='right', va='top', fontsize=10)
fig.text(0.125, 0.89, time_formatted, ha='left', va='top', fontsize=12)
fig.text(0.77, 0.89, 'WRF Model with 10m Surface Winds (m/s)',
         ha='right', va='top', fontsize=10)

# Adjust layout
plt.subplots_adjust(top=1, bottom=0)

# Define output folder
output_dir = Path('figures')
output_dir.mkdir(parents=True, exist_ok=True)
fig_path = output_dir / f'hurricane_ian_center_{t}.png'

# Save and show the plot
fig.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
# plt.show()
t_end = time.time()
print(
    f"Plot saved to {fig_path}. Elapsed time: {t_end - t_start:.2f} seconds.")
# ---------------------------------------------------------------------------------------
# 8. Print results
# ---------------------------------------------------------------------------------------
print("\nFinal center estimates:")
print(f"  Pressure minimum center : {center_lat:.2f}°, {center_lon:.2f}°")
print(f"  Eyewall-refined center   : {refined_lat:.2f}°, {refined_lon:.2f}°")
aeqd_refined = Proj(proj='aeqd', lat_0=center_lat,
                    lon_0=center_lon, datum='WGS84')
x_off, y_off = Transformer.from_proj(
    'epsg:4326', aeqd_refined, always_xy=True).transform(refined_lon, refined_lat)
offset_km = np.sqrt(x_off**2 + y_off**2) / 1000
print(f"  Offset distance          : ~{offset_km:.1f} km")
