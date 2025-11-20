#!/usr/bin/env python3
"""
plt_crude_vertical_transport_robust.py
--------------------------------------
Crude vertical enthalpy transport: Fz = (LH+HFX) * mean(W)
* Skips any timestep that cannot be read
* Safe unstagger that works whether the variable is staggered or not
"""

import numpy as np
import netCDF4 as nc
from pathlib import Path
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy.stats import binned_statistic
from pyproj import Proj, Transformer
import logging
from datetime import datetime
import time

# --------------------------------------------------------------
# Logging
# --------------------------------------------------------------
logging.basicConfig(
    filename='crude_vertical_transport_robust.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

# --------------------------------------------------------------
# 1. Safe read
# --------------------------------------------------------------


def safe_read(var, t, name=""):
    try:
        return var[t]
    except Exception as e:
        logging.warning(f"Failed to read {name} at t={t}: {e}")
        print(f"WARNING: Failed to read {name} at t={t}: {e}")
        return None

# --------------------------------------------------------------
# 2. Robust unstagger
# --------------------------------------------------------------


def unstagger(var, dim, mass_shape):
    """
    Unstagger along `dim` (0 = south-north, 1 = west-east).
    `mass_shape` = shape of the *mass* grid (e.g. (south_north, west_east)).
    If the input is already the same size as `mass_shape`, return it unchanged.
    """
    if var is None:
        return None

    # Expected staggered size = mass size + 1 in the chosen dimension
    expected_stag = list(mass_shape)
    expected_stag[dim] += 1

    if var.shape[-2:] == tuple(expected_stag):          # really staggered
        if dim == 0:   # south-north
            return 0.5 * (var[..., :-1, :] + var[..., 1:, :])
        else:          # west-east
            return 0.5 * (var[..., :-1] + var[..., 1:, :])
    else:                                               # already mass-grid
        return var

# --------------------------------------------------------------
# 3. Center refinement (unchanged)
# --------------------------------------------------------------


def compute_annulus_mean(args):
    i, j, lat, lon, u10, v10, Rmax, annulus = args
    try:
        clat, clon = lat[i, j], lon[i, j]
        aeqd = Proj(proj='aeqd', lat_0=clat, lon_0=clon, datum='WGS84')
        tr = Transformer.from_proj('epsg:4326', aeqd, always_xy=True)
        x, y = tr.transform(lon, lat)
        r = np.hypot(x, y) / 1e3
        theta = np.arctan2(y, x)

        mask = r <= Rmax
        tan = -u10 * np.sin(theta) + v10 * np.cos(theta)
        tan, r = np.where(mask, tan, np.nan), np.where(mask, r, np.nan)

        bins = np.linspace(0, Rmax, 60)
        mean_tan, _, _ = binned_statistic(r.ravel(), tan.ravel(),
                                          statistic='mean', bins=bins)
        rmid = 0.5 * (bins[:-1] + bins[1:])
        valid = ~np.isnan(mean_tan)
        if not valid.any():
            return -np.inf, clat, clon
        rmw = rmid[valid][mean_tan[valid].argmax()]

        ann_mask = (r >= rmw - annulus) & (r < rmw + annulus)
        if ann_mask.sum() == 0:
            return -np.inf, clat, clon
        return np.nanmean(tan[ann_mask]), clat, clon
    except Exception as e:
        logging.error(f"annulus error i,j={i},{j}: {e}")
        return -np.inf, clat, clon


# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------
if __name__ == '__main__':
    mp.freeze_support()
    t0 = time.time()

    # ------------------- USER SETTINGS -------------------
    wrf_file = Path('../data/wrfout_d03_2022-09-26_Ian2022_UNCPL.nc')
    start_step = 50
    Rmax_km = 250
    annulus_km = 15
    box_half = 20
    radial_bins = 100
    # ----------------------------------------------------

    ds = nc.Dataset(wrf_file)

    # Static fields (always safe)
    try:
        lat = ds['XLAT'][0]
        lon = ds['XLONG'][0]
        mass_sn, mass_we = lat.shape               # mass-grid size
    except Exception as e:
        logging.error(f"Cannot read lat/lon: {e}")
        raise

    ntimes = ds['Times'].shape[0]

    for t in range(start_step, ntimes):
        logging.info(f'=== TRYING TIMESTEP {t} ===')
        print(f'\n--- Trying timestep {t} ---')

        # ------------------------------------------------------
        # 1. Read required variables safely
        # ------------------------------------------------------
        LH = safe_read(ds.variables.get('LH'),   t, 'LH')
        HFX = safe_read(ds.variables.get('HFX'),  t, 'HFX')
        if LH is None or HFX is None:
            print(f"Skipping t={t}: missing LH or HFX")
            continue
        surf_flux = LH + HFX

        u10 = safe_read(ds.variables.get('U10'), t, 'U10')
        v10 = safe_read(ds.variables.get('V10'), t, 'V10')
        if u10 is None or v10 is None:
            print(f"Skipping t={t}: missing U10/V10")
            continue

        W_stag = safe_read(ds.variables.get('W'), t, 'W')
        if W_stag is None:
            print(f"Skipping t={t}: missing W")
            continue

        psfc = safe_read(ds.variables.get('PSFC'), t, 'PSFC')
        if psfc is None:
            print(f"Skipping t={t}: missing PSFC")
            continue

        # ------------------------------------------------------
        # 2. Unstagger U10/V10 (only if really staggered)
        # ------------------------------------------------------
        u10 = unstagger(u10, dim=1, mass_shape=(mass_sn, mass_we))
        v10 = unstagger(v10, dim=0, mass_shape=(mass_sn, mass_we))

        # ------------------------------------------------------
        # 3. Unstagger W (vertical velocity) – safe
        # ------------------------------------------------------
        W = unstagger(W_stag, dim=0, mass_shape=(mass_sn, mass_we))
        if W is None:
            print(f"Skipping t={t}: W unstagger failed")
            continue

        # ------------------------------------------------------
        # 4. Time string
        # ------------------------------------------------------
        try:
            time_bytes = ds['Times'][t]
            time_str = ''.join(b.decode('utf-8') for b in time_bytes)
            tdt = datetime.strptime(time_str, '%Y-%m-%d_%H:%M:%S')
            label_time = tdt.strftime('%H:%Mz %d %b %Y')
        except Exception:
            label_time = f"t={t}"

        # ------------------------------------------------------
        # 5. Find storm centre
        # ------------------------------------------------------
        try:
            i0, j0 = np.unravel_index(psfc.argmin(), psfc.shape)
            lat0, lon0 = float(lat[i0, j0]), float(lon[i0, j0])

            i_lo = max(0, i0-box_half)
            i_hi = min(mass_sn, i0+box_half+1)
            j_lo = max(0, j0-box_half)
            j_hi = min(mass_we, j0+box_half+1)

            candidates = [(i, j, lat, lon, u10, v10, Rmax_km, annulus_km)
                          for i in range(i_lo, i_hi) for j in range(j_lo, j_hi)]

            with mp.Pool(min(mp.cpu_count(), len(candidates))) as pool:
                results = pool.map(compute_annulus_mean, candidates)

            best_tan, lat_c, lon_c = max(results, key=lambda x: x[0])
            if best_tan == -np.inf:
                raise ValueError("center refinement failed")
        except Exception as e:
            logging.warning(f"Center finding failed at t={t}: {e}")
            print(f"Skipping t={t}: centre failed")
            continue

        # ------------------------------------------------------
        # 6. Storm-centred radius
        # ------------------------------------------------------
        try:
            aeqd = Proj(proj='aeqd', lat_0=lat_c, lon_0=lon_c, datum='WGS84')
            tr = Transformer.from_proj('epsg:4326', aeqd, always_xy=True)
            X, Y = tr.transform(lon, lat)
            R = np.hypot(X, Y) / 1e3                     # km
        except Exception as e:
            logging.warning(f"Projection failed at t={t}: {e}")
            continue

        # ------------------------------------------------------
        # 7. Azimuthal averages
        # ------------------------------------------------------
        r_edges = np.linspace(0, Rmax_km, radial_bins + 1)
        r_cent = 0.5 * (r_edges[:-1] + r_edges[1:])

        def az_mean(field2d):
            stat, _, _ = binned_statistic(R.ravel(), field2d.ravel(),
                                          statistic='mean', bins=r_edges)
            return np.nan_to_num(stat)

        surf_az = az_mean(surf_flux)
        W_mean_2d = W.mean(axis=0)                     # height-averaged W
        W_az = az_mean(W_mean_2d)

        # ------------------------------------------------------
        # 8. Crude vertical transport
        # ------------------------------------------------------
        crude_flux = surf_az * W_az                     # W m^-2

        # ------------------------------------------------------
        # 9. Plot
        # ------------------------------------------------------
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                           constrained_layout=True)

            ax1.plot(r_cent, surf_az, color='tab:green', lw=2)
            ax1.set_xlabel('Radius (km)')
            ax1.set_ylabel('Surface Flux (W m^-2)')
            ax1.set_title(f'Surface LH+HFX\n{label_time}')
            ax1.grid(True, ls=':')

            ax2.plot(r_cent, crude_flux, color='tab:red', lw=2)
            ax2.set_xlabel('Radius (km)')
            ax2.set_ylabel('Crude Fz (W m^-2)')
            ax2.set_title('Crude Upward Transport\nFz = (LH+HFX)×W')
            ax2.grid(True, ls=':')

            fig.suptitle(f'Hurricane IAN – t={t:03d} – '
                         f'Center {lat_c:.2f}°N {lon_c:.2f}°E',
                         fontsize=14)

            outdir = Path('figures_crude_robust')
            outdir.mkdir(exist_ok=True)
            figname = outdir / f'crude_t{t:04d}.png'
            fig.savefig(figname, dpi=300, bbox_inches='tight')
            plt.close(fig)

            logging.info(f'SUCCESS: Saved {figname}')
            print(f"SUCCESS: Saved {figname}")

        except Exception as e:
            logging.error(f"Plot failed at t={t}: {e}")
            print(f"Plot failed at t={t}: {e}")

    ds.close()
    elapsed = time.time() - t0
    logging.info(f'Finished. Elapsed {elapsed:.1f}s')
    print(f"\nDone!  Figures are in: figures_crude_robust/")
