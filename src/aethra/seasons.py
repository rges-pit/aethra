"""Season splitting and per-season scan."""

import numpy as np

from .detection import detect_bump, flatness_chi2
from .variability import is_non_flat_lightcurve

__all__ = ["split_into_seasons", "analyze_season_scan"]


def split_into_seasons(time, gap_days=100):
    time = np.asarray(time, dtype=float)
    if len(time) == 0:
        return np.array([], dtype=int)
    order    = np.argsort(time)
    t_sorted = time[order]
    labels_sorted = np.zeros(len(t_sorted), dtype=int)
    season_id = 1
    for i in range(1, len(t_sorted)):
        if t_sorted[i] - t_sorted[i - 1] > gap_days:
            season_id += 1
        labels_sorted[i] = season_id
    labels = np.empty(len(time), dtype=int)
    labels[order] = labels_sorted
    return labels


def analyze_season_scan(time, mags, mag_err, min_points=10):
    time    = np.asarray(time,    dtype=float)
    mags    = np.asarray(mags,    dtype=float)
    mag_err = np.asarray(mag_err, dtype=float)

    valid = np.isfinite(time) & np.isfinite(mags) & np.isfinite(mag_err)
    time, mags, mag_err = time[valid], mags[valid], mag_err[valid]

    empty = {
        "is_non_flat": False, "baseline_mag": np.nan, "peak_mag": np.nan,
        "bump_flag": False,   "bump_snr": np.nan,     "best_window": np.nan,
        "chi2_flat": np.nan,  "dof_flat": np.nan,     "chi2_red_flat": np.nan,
        "t0_guess_raw": np.nan,
    }

    if len(mags) < min_points or np.ptp(time) <= 0:
        return empty

    positive_err = mag_err[mag_err > 0]
    if len(positive_err) == 0:
        return empty

    err_floor = max(np.median(positive_err) * 0.1, 1e-3)
    mag_err   = np.where(mag_err <= 0, err_floor, mag_err)

    flux     = 10 ** (-0.4 * mags)
    flux_err = 0.4 * np.log(10) * flux * mag_err
    chi2_flat, dof_flat, chi2_red_flat = flatness_chi2(flux, flux_err)

    outlier_flag = is_non_flat_lightcurve(mags, sigma_thresh=2.0, n_consecutive=2)
    is_non_flat  = outlier_flag or (np.isfinite(chi2_red_flat) and chi2_red_flat > 1.5)

    bump_flag, best_window, bump_snr = detect_bump(
        time, mags, mag_err,
        window_days=(2, 5, 10, 20, 50), snr_threshold=2.5, n_consecutive=2,
    )

    t0_guess_raw = time[np.argmin(mags)] if len(time) > 0 else np.nan

    return {
        "is_non_flat":   is_non_flat,
        "baseline_mag":  float(np.median(mags)),
        "peak_mag":      float(np.min(mags)),
        "bump_flag":     bump_flag,
        "bump_snr":      bump_snr,
        "best_window":   best_window,
        "chi2_flat":     chi2_flat,
        "dof_flat":      dof_flat,
        "chi2_red_flat": chi2_red_flat,
        "t0_guess_raw":  t0_guess_raw,
    }
