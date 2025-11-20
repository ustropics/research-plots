# ---------------------------------------------------------------------------------------
# plt_vertical_enthalpy_flux.py
#   Computes and plots vertically integrated vertical enthalpy flux (ρ w h)
#   Uses same storm-center refinement as plt_efx.py
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
import multiprocessing
import logging

# --- Physical constants ---
cp = 1004.0      # J/kg/K
Lv = 2.5e6       # J/kg
Rd = 287.0       # J/kg/K
g = 9.81         # m/s^2

# ---------------------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------------------


def unstagger(var, axis):
    return 0.5 * (var.take(indices=range(var.shape[axis]-1), axis=axis) +
                  var.take(indices=range(1, var.shape[axis]), axis=axis))


def compute_annulus_mean(args):
    i, j, lat, lon, u10, v10, search_radius_km, annulus_width_km = args
    try:
        cand_lat, cand_lon = lat[i, j], lon[i, j]
        aeqd = Proj(proj='aeqd', lat_0=cand_lat, lon_0=cand_lon, datum='WGS84')
        transformer = Transformer.from_proj('epsg:4326', aeqd, always_xy=True)
        x, y = transformer.transform(lon, lat)
        r = np.sqrt(x**2 + y**2) / 1000.0
        theta = np.arctan2(y, x)
        mask = r <= search_radius_km
        u10_m, v10_m = np.where(mask, u10, np.nan), np.where(mask, v10, np.nan)
        tangential = -u10_m*np.sin(theta) + v10_m*np.cos(theta)
        bins = np.linspace(0, search_radius_km, 50)
        mean_tang, edges, _ = binned_statistic(r[mask].flatten(), tangential[mask].flatten(),
                                               statistic='mean', bins=bins)
        r_mid = 0.5*(edges[:-1] + edges[1:])
        if np.all(np.isnan(mean_tang)):
            return -np.inf, cand_lat, cand_lon
        rmw = r_mid[np.nanargmax(mean_tang)]
        annulus_mask = (r >= rmw-15) & (r < rmw+15)
        if np.sum(annulus_mask) == 0:
            return -np.inf, cand_lat, cand_lon
        annulus_mean = np.nanmean(tangential[annulus_mask])
        return annulus_mean, cand_lat, cand_lon
    except Exception as e:
        logging.error(f"Error annulus_mean ({i},{j}): {e}")
        return -np.inf, cand_lat, cand_lon


# ---------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------
if __name__ == '__main__':
    multiprocessing.freeze_support()
    logging.basicConfig(filename='vertical_enthalpy_flux.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    data_path = Path('../data/wrfout_d03_2022-09-26_Ian2022_UNCPL.nc')
    ds = nc.Dataset(data_path)

    lat = ds.variables['XLAT'][0, :, :]
    lon = ds.variables['XLONG'][0, :, :]
    num_timesteps = ds.variables['Times'].shape[0]
    start_timestep = 50

    logging.info(f"Processing timesteps {start_timestep}–{num_timesteps-1}")

    for t in range(start_timestep, num_timesteps):
        try:
            logging.info(f"Timestep {t}")
            print(f"Processing timestep {t}")

            # --- Find center (same as surface script) ---
            psfc = ds.variables['PSFC'][t, :, :]
            imin_psfc = np.unravel_index(np.argmin(psfc), psfc.shape)
            center_lat, center_lon = float(
                lat[imin_psfc]), float(lon[imin_psfc])
            u10, v10 = ds.variables['U10'][t, :,
                                           :], ds.variables['V10'][t, :, :]

            # refine using annulus method
            search_box = 20
            i_min, j_min = imin_psfc
            i_range = range(max(0, i_min-search_box),
                            min(lat.shape[0], i_min+search_box))
            j_range = range(max(0, j_min-search_box),
                            min(lat.shape[1], j_min+search_box))
            candidates = [(i, j, lat, lon, u10, v10, 200, 15)
                          for i in i_range for j in j_range]
            with multiprocessing.Pool(min(multiprocessing.cpu_count(), len(candidates))) as pool:
                results = pool.map(compute_annulus_mean, candidates)
            ann_means, lats_c, lons_c = zip(*results)
            refined_lat = lats_c[np.argmax(ann_means)]
            refined_lon = lons_c[np.argmax(ann_means)]

            # --- 3D fields ---
            P = ds.variables['P'][t, :, :, :]
            PB = ds.variables['PB'][t, :, :, :]
            T_p = ds.variables['T'][t, :, :, :]
            QV = ds.variables['QVAPOR'][t, :, :, :]
            W = ds.variables['W'][t, :, :, :]
            PH = ds.variables['PH'][t, :, :, :]
            PHB = ds.variables['PHB'][t, :, :, :]
            z = (PH + PHB) / g  # height (m)

            # Temperature in K
            p = P + PB
            T_abs = (T_p + 300.0) * (p / 1e5) ** 0.286
            rho = p / (Rd * T_abs)
            h = cp * T_abs + Lv * QV + g * z
            Fz = rho * W * h  # W/m^2

            # integrate vertically
            dz = np.gradient(z, axis=0)
            Fz_int = np.nansum(Fz * dz, axis=0)

            # --- Plot ---
            time_bytes = ds.variables['Times'][t]
            time_str = ''.join([b.decode('utf-8') for b in time_bytes])
            time_dt = datetime.strptime(time_str, '%Y-%m-%d_%H:%M:%S')
            time_fmt = time_dt.strftime('%H:%Mz, %b %d, %Y')

            fig, ax = plt.subplots(figsize=(12, 10),
                                   subplot_kw={'projection': ccrs.PlateCarree()})
            ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()])
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=':')

            im = ax.contourf(lon, lat, Fz_int/1e6, levels=np.linspace(-2, 2, 21),
                             cmap='coolwarm', transform=ccrs.PlateCarree())
            cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
            cbar.set_label(
                'Vertically Integrated Vertical Enthalpy Flux (MW/m²)')

            ax.plot(refined_lon, refined_lat, 'ko', markersize=6)
            fig.text(0.13, 0.92, f'Hurricane IAN – {time_fmt}', fontsize=12)
            fig.text(
                0.75, 0.92, f'Center: ({refined_lat:.2f}, {refined_lon:.2f})', fontsize=10)

            out_dir = Path('figures_vertical_enthalpy')
            out_dir.mkdir(exist_ok=True)
            out_file = out_dir / f'vertical_enthalpy_flux_{t:03d}.png'
            plt.savefig(out_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Timestep {t} saved to {out_file}")

        except Exception as e:
            logging.error(f"Timestep {t} error: {e}")
            print(f"Error processing timestep {t}: {e}")
            continue

    ds.close()
    print("All done!")
