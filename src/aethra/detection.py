"""Outlier, flatness, and bump-detection routines."""

import numpy as np
import pandas as pd

__all__ = [
    "has_consecutive_outliers",
    "flatness_chi2",
    "detect_bump",
    "recurrent_bump_veto",
]


def has_consecutive_outliers(threshold_mask, n_consecutive=3):
    count = 0
    for val in threshold_mask:
        if val:
            count += 1
            if count >= n_consecutive:
                return True
        else:
            count = 0
    return False


def flatness_chi2(flux, flux_err):
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)

    valid = np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)
    flux = flux[valid]
    flux_err = flux_err[valid]

    if len(flux) < 2:
        return np.nan, np.nan, np.nan

    w = 1.0 / flux_err**2
    wsum = np.sum(w)
    if not np.isfinite(wsum) or wsum <= 0:
        return np.nan, np.nan, np.nan

    c = np.sum(w * flux) / wsum
    chi2 = np.sum(w * (flux - c) ** 2)
    dof = len(flux) - 1
    chi2_red = chi2 / dof if dof > 0 else np.nan

    return chi2, dof, chi2_red


def detect_bump(time, mags, mag_err, window_days=(1, 2, 5, 10, 20, 50),
                snr_threshold=3.0, n_consecutive=3):
    time = np.asarray(time, dtype=float)
    mags = np.asarray(mags, dtype=float)
    mag_err = np.asarray(mag_err, dtype=float)

    valid = np.isfinite(time) & np.isfinite(mags) & np.isfinite(mag_err) & (mag_err > 0)
    time, mags, mag_err = time[valid], mags[valid], mag_err[valid]

    if len(mags) < 5:
        return False, np.nan, np.nan

    dt = np.diff(time)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return False, np.nan, np.nan

    cadence = np.median(dt)

    flux = 10 ** (-0.4 * mags)
    flux_err = 0.4 * np.log(10) * flux * mag_err

    valid_flux = np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)
    flux = flux[valid_flux]
    flux_err = flux_err[valid_flux]

    if len(flux) < 5:
        return False, np.nan, np.nan

    baseline_flux = np.percentile(flux, 30)
    weighted_excess = (flux - baseline_flux) / flux_err

    best_snr = -np.inf
    best_window_days = np.nan

    for w_days in window_days:
        w_pts = max(n_consecutive, int(round(w_days / cadence)))
        if w_pts >= len(flux):
            continue

        rolling_snr = (
            pd.Series(weighted_excess)
            .rolling(w_pts, center=True, min_periods=w_pts)
            .sum()
            .to_numpy()
            / np.sqrt(w_pts)
        )

        if len(rolling_snr) == 0 or not np.any(np.isfinite(rolling_snr)):
            continue

        peak_snr = np.nanmax(rolling_snr)
        if np.isfinite(peak_snr) and peak_snr > best_snr:
            best_snr = peak_snr
            best_window_days = float(w_days)

        mask = rolling_snr > snr_threshold
        if has_consecutive_outliers(mask, n_consecutive=n_consecutive):
            return True, float(w_days), float(peak_snr)

    return False, best_window_days, (best_snr if np.isfinite(best_snr) else np.nan)


def recurrent_bump_veto(
    obj_df_primary,
    best_season,
    best_bump_snr,
    time_col="bjd",
    mag_col="mag",
    err_col="mag_err",
    season_col="season_id",
    min_points=10,
    min_other_bump_seasons=1,
):
    other_hits = []

    for season_id, season_df in obj_df_primary.groupby(season_col):
        if season_id == best_season:
            continue
        if len(season_df) < min_points:
            continue

        bump_flag, best_window, bump_snr = detect_bump(
            season_df[time_col].to_numpy(),
            season_df[mag_col].to_numpy(),
            season_df[err_col].to_numpy(),
            window_days=(5, 10, 20, 50),
            snr_threshold=4.0,
            n_consecutive=3,
        )

        if not bump_flag or not np.isfinite(bump_snr):
            continue

        strong_enough = bump_snr >= max(25.0, 0.25 * best_bump_snr)
        broad_enough = np.isfinite(best_window) and (best_window >= 5)

        if strong_enough and broad_enough:
            other_hits.append((season_id, bump_snr, best_window))

    veto = len(other_hits) >= min_other_bump_seasons
    return veto, other_hits
