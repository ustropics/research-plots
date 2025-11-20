# ---------------------------------------------------------------------------------------
# plt_center_enthalpy_prl.py
#   Same logic as plt_center_prl.py but plots TOTAL ENTHALPY FLUX (LH + HFX)
#   ALL ERRORS FIXED: 'nc.Dataset', proper imports, clean logic
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
import multiprocessing
import logging

# Set up logging
logging.basicConfig(
    filename='hurricane_ian_enthalpy_processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Function: Unstagger WRF variables


def unstagger(var, axis):
    return 0.5 * (var.take(indices=range(var.shape[axis]-1), axis=axis) +
                  var.take(indices=range(1, var.shape[axis]), axis=axis))

# Function to compute annulus mean for candidate center


def compute_annulus_mean(args):
    i, j, lat, lon, u10, v10, search_radius_km, annulus_width_km = args
    try:
        cand_lat = lat[i, j]
        cand_lon = lon[i, j]

        aeqd = Proj(proj='aeqd', lat_0=cand_lat, lon_0=cand_lon, datum='WGS84')
        transformer = Transformer.from_proj('epsg:4326', aeqd, always_xy=True)
        x, y = transformer.transform(lon, lat)

        r = np.sqrt(x**2 + y**2) / 1000.0  # km
        theta = np.arctan2(y, x)

        mask = r <= search_radius_km
        r_masked = np.where(mask, r, np.nan)
        theta_masked = np.where(mask, theta, np.nan)
        u10_masked = np.where(mask, u10, np.nan)
        v10_masked = np.where(mask, v10, np.nan)

        tangential = -u10_masked * \
            np.sin(theta_masked) + v10_masked * np.cos(theta_masked)

        bins = np.linspace(0, search_radius_km, 50)
        mean_tang, bin_edges, _ = binned_statistic(
            r_masked.flatten(), tangential.flatten(), statistic='mean', bins=bins)
        r_mid = (bin_edges[:-1] + bin_edges[1:]) / 2

        valid = ~np.isnan(mean_tang)
        if not np.any(valid):
            return -np.inf, cand_lat, cand_lon
        rmw = r_mid[valid][np.argmax(mean_tang[valid])]

        annulus_mask = (r_masked >= rmw -
                        annulus_width_km) & (r_masked < rmw + annulus_width_km)
        if np.sum(annulus_mask) == 0:
            return -np.inf, cand_lat, cand_lon
        annulus_mean = np.nanmean(tangential[annulus_mask])

        return annulus_mean, cand_lat, cand_lon
    except Exception as e:
        logging.error(f"Error in compute_annulus_mean for i={i}, j={j}: {e}")
        return -np.inf, cand_lat, cand_lon


# ---------------------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------------------
if __name__ == '__main__':
    multiprocessing.freeze_support()
    t_start = time.time()
    logging.info("Script started at: %s",
                 datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # Load WRF data
    logging.info("Loading WRF data...")
    data_path = Path('../data/wrfout_d03_2022-09-26_Ian2022_UNCPL.nc')
    ds = nc.Dataset(data_path)  # ← FIXED: 'nc', not 'nw'

    # Get lat/lon (static)
    lat = ds.variables['XLAT'][0, :, :]
    lon = ds.variables['XLONG'][0, :, :]

    num_timesteps = ds.variables['Times'].shape[0]
    start_timestep = 50
    logging.info(f"Processing timesteps {start_timestep} to {num_timesteps-1}")

    # <<< CHANGE >>> Define enthalpy flux (LH + HFX)
    lh_var = 'LH'
    hfx_var = 'HFX'
    cmap_flux = 'twilight'
    flux_levels = np.linspace(0, 2000, 21)  # Adjust after checking data
    flux_label = 'Total Surface Enthalpy Flux (W/m²)'

    for t in range(start_timestep, num_timesteps):
        try:
            logging.info(f"Processing timestep {t}")
            print(f"Processing timestep {t}")

            # ---------------------------------------------------------------------------------------
            # 3. Read U10/V10 for center finding
            # ---------------------------------------------------------------------------------------
            u10 = ds.variables['U10'][t, :, :]
            v10 = ds.variables['V10'][t, :, :]

            if 'west_east_stag' in ds.variables['U10'].dimensions:
                u10 = unstagger(u10, axis=1)
            if 'south_north_stag' in ds.variables['V10'].dimensions:
                v10 = unstagger(v10, axis=0)

            u10 = np.array(u10)
            v10 = np.array(v10)

            # ---------------------------------------------------------------------------------------
            # 4. Initial center: minimum surface pressure
            # ---------------------------------------------------------------------------------------
            psfc = ds.variables['PSFC'][t, :, :]
            imin_psfc = np.unravel_index(np.argmin(psfc), psfc.shape)
            center_lat = float(lat[imin_psfc])
            center_lon = float(lon[imin_psfc])
            logging.info(
                f"Timestep {t} - Initial center: {center_lat:.2f}°, {center_lon:.2f}°")
            print(
                f"Timestep {t} - Initial center: {center_lat:.2f}°, {center_lon:.2f}°")

            # ---------------------------------------------------------------------------------------
            # 5. Refine center using max tangential wind
            # ---------------------------------------------------------------------------------------
            search_radius_km = 200
            annulus_width_km = 15
            search_box = 20

            i_min, j_min = imin_psfc
            i_range = range(max(0, i_min - search_box),
                            min(lat.shape[0], i_min + search_box + 1))
            j_range = range(max(0, j_min - search_box),
                            min(lat.shape[1], j_min + search_box + 1))

            candidates = [(i, j, lat, lon, u10, v10, search_radius_km, annulus_width_km)
                          for i in i_range for j in j_range]

            num_processes = min(multiprocessing.cpu_count(), len(candidates))
            logging.info(
                f"Timestep {t} - Using {num_processes} processes for {len(candidates)} candidates")
            print(
                f"Timestep {t} - Using {num_processes} processes for {len(candidates)} candidates")

            with multiprocessing.Pool(processes=num_processes) as pool:
                results = pool.map(compute_annulus_mean, candidates)

            max_annulus_mean = -np.inf
            refined_lat = center_lat
            refined_lon = center_lon
            for annulus_mean, cand_lat, cand_lon in results:
                if annulus_mean > max_annulus_mean:
                    max_annulus_mean = annulus_mean
                    refined_lat = cand_lat
                    refined_lon = cand_lon

            logging.info(
                f"Timestep {t} - Refined center: {refined_lat:.2f}°, {refined_lon:.2f}°")
            print(
                f"Timestep {t} - Refined center: {refined_lat:.2f}°, {refined_lon:.2f}°")

            # ---------------------------------------------------------------------------------------
            # 6. Compute radial/tangential wind (for rings)
            # ---------------------------------------------------------------------------------------
            aeqd_refined = Proj(proj='aeqd', lat_0=refined_lat,
                                lon_0=refined_lon, datum='WGS84')
            transformer_refined = Transformer.from_proj(
                'epsg:4326', aeqd_refined, always_xy=True)
            x_ref, y_ref = transformer_refined.transform(lon, lat)

            # ---------------------------------------------------------------------------------------
            # 7. <<< CHANGE >>> COMPUTE TOTAL ENTHALPY FLUX (LH + HFX)
            # ---------------------------------------------------------------------------------------
            lh = ds.variables[lh_var][t, :, :]
            hfx = ds.variables[hfx_var][t, :, :]
            lh = np.ma.filled(lh, np.nan)
            hfx = np.ma.filled(hfx, np.nan)
            enthalpy = lh + hfx  # Total enthalpy flux

            max_enthalpy = np.nanmax(enthalpy)
            min_pressure = np.min(psfc) / 100

            # ---------------------------------------------------------------------------------------
            # 8. Plotting
            # ---------------------------------------------------------------------------------------
            time_bytes = ds.variables['Times'][t]
            time_str = ''.join([byte.decode('utf-8') for byte in time_bytes])
            time_dt = datetime.strptime(time_str, '%Y-%m-%d_%H:%M:%S')
            time_formatted = time_dt.strftime('%H:%Mz, %b %d, %Y')

            fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={
                                   'projection': ccrs.PlateCarree()})
            ax.set_extent([lon.min(), lon.max(), lat.min(),
                          lat.max()], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=':')

            # Plot enthalpy
            im = ax.contourf(lon, lat, enthalpy, levels=flux_levels,
                             cmap=cmap_flux, transform=ccrs.PlateCarree())

            # Optional wind vectors
            ax.quiver(lon[::20, ::20], lat[::20, ::20], u10[::20, ::20],
                      v10[::20, ::20], scale=1500, transform=ccrs.PlateCarree())

            # Radial rings
            radius_25_miles = 25 / 69
            radius_50_miles = 50 / 69
            radius_100_miles = 100 / 69
            radius_150_miles = 150 / 69
            radius_200_miles = 200 / 69

            rad_25 = plt.Circle((refined_lon, refined_lat), radius_25_miles,
                                color="#ff32f5", fill=False, linewidth=2, alpha=0.6, label='25 miles')
            rad_50 = plt.Circle((refined_lon, refined_lat), radius_50_miles,
                                color="#c00000", fill=False, linewidth=2, alpha=0.6, label='50 miles')
            rad_100 = plt.Circle((refined_lon, refined_lat), radius_100_miles,
                                 color="#ec4b00", fill=False, linewidth=2, alpha=0.6, label='100 miles')
            rad_150 = plt.Circle((refined_lon, refined_lat), radius_150_miles,
                                 color="#EED600", fill=False, linewidth=2, alpha=0.6, label='150 miles')
            rad_200 = plt.Circle((refined_lon, refined_lat), radius_200_miles,
                                 color="#12D400", fill=False, linewidth=2, alpha=0.6, label='200 miles')

            for rad in [rad_25, rad_50, rad_100, rad_150, rad_200]:
                ax.add_patch(rad)

            ax.plot(refined_lon, refined_lat, 'o', color="#A70075",
                    markersize=10, label='Eyewall-Refined Center')
            plt.legend(handles=[rad_25, rad_50, rad_100,
                       rad_150, rad_200], loc='upper right')

            # Colorbar
            cbar = fig.colorbar(im, ax=ax, orientation='vertical',
                                pad=0.02, aspect=20, shrink=0.7)
            cbar.set_label(flux_label, fontsize=10)

            # Text
            fig.text(0.125, 0.91, 'Hurricane IAN',
                     ha='left', va='top', fontsize=12)
            fig.text(0.77, 0.91,
                     f'Lat: {refined_lat:.2f} | Lon: {refined_lon:.2f} | '
                     f'Max Enthalpy: {max_enthalpy:.1f} W/m² | Min P: {min_pressure:.1f} hPa',
                     ha='right', va='top', fontsize=10)
            fig.text(0.125, 0.89, time_formatted,
                     ha='left', va='top', fontsize=12)
            fig.text(0.77, 0.89, 'WRF Model - Total Surface Enthalpy Flux (W/m²)',
                     ha='right', va='top', fontsize=10)

            plt.subplots_adjust(top=1, bottom=0)

            output_dir = Path('figures_enthalpy')
            output_dir.mkdir(parents=True, exist_ok=True)
            fig_path = output_dir / f'hurricane_ian_enthalpy_{t}.png'

            fig.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)
            logging.info(f"Timestep {t} - Plot saved to {fig_path}")
            print(f"Timestep {t} - Plot saved to {fig_path}")

            # ---------------------------------------------------------------------------------------
            # 9. Log center offset
            # ---------------------------------------------------------------------------------------
            aeqd_refined = Proj(proj='aeqd', lat_0=center_lat,
                                lon_0=center_lon, datum='WGS84')
            x_off, y_off = Transformer.from_proj('epsg:4326', aeqd_refined, always_xy=True).transform(
                refined_lon, refined_lat)
            offset_km = np.sqrt(x_off**2 + y_off**2) / 1000

            result_str = (f"Timestep {t} - Final center estimates:\n"
                          f"  Pressure minimum center : {center_lat:.2f}°, {center_lon:.2f}°\n"
                          f"  Eyewall-refined center   : {refined_lat:.2f}°, {refined_lon:.2f}°\n"
                          f"  Offset distance          : ~{offset_km:.1f} km")
            logging.info(result_str)
            print(result_str)

        except Exception as e:
            logging.error(f"Error processing timestep {t}: {e}")
            print(f"Error processing timestep {t}: {e}")
            continue

    t_end = time.time()
    logging.info(
        f"Script completed. Total elapsed time: {t_end - t_start:.2f} seconds")
    print(
        f"Script completed. Total elapsed time: {t_end - t_start:.2f} seconds")
    ds.close()
