"""
orbit_analysis_kepler.py
------------------------
Manoeuvre detection using Keplerian element deltas from CelesTrak orbital history.

Data source : https://celestrak.org  (no account required)
Method      : Propagate each epoch's Keplerian elements forward to the next epoch,
              compare predicted vs actual position, flag large SMA/eccentricity
              step-changes as manoeuvres, cross-check with Kp space weather index.

Usage:
    python orbit_analysis_kepler.py
    or import and call analyse_maneuvers_kepler(catnr, data_dir, output_dir, ...)
"""

import re
import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from datetime import datetime


# ── 1. Orbit Data ─────────────────────────────────────────────────────────────

def get_orbit_data(catnr: int) -> pd.DataFrame:
    """Fetches orbital element history from CelesTrak graph data."""
    url = "https://celestrak.org/NORAD/elements/graph-orbit-data.php"
    response = requests.get(url, params={"CATNR": catnr}, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()

    match = re.search(r'var plotData = "(.+?)"', response.text, re.DOTALL)
    if not match:
        raise ValueError(f"No plotData found for CATNR={catnr}")

    df = pd.read_csv(StringIO(match.group(1).replace('|', '\n')))
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def load_orbit_data(catnr: int, data_dir: str = "./data", max_age_days: int = 1) -> dict[str, pd.DataFrame]:
    """
    Loads orbital data from local CSV if fresh, otherwise fetches from CelesTrak.

    Args:
        catnr:        NORAD catalog number
        data_dir:     Folder to check/save CSV files
        max_age_days: Refresh if local data is older than this many days

    Returns:
        Dictionary of DataFrames keyed by element name, plus 'full' for the raw df
    """
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, f"orbit_{catnr}.csv")

    if os.path.exists(filepath):
        df = pd.read_csv(filepath, parse_dates=['Date'])
        age_days = (datetime.now() - df['Date'].max().to_pydatetime().replace(tzinfo=None)).days
        if age_days > max_age_days:
            print(f"Local orbit file is {age_days} days old. Refreshing from CelesTrak...")
            df = get_orbit_data(catnr)
            df.to_csv(filepath, index=False)
            print(f"Updated {filepath}")
        else:
            print(f"Loaded local orbit file (latest: {df['Date'].max()})")
    else:
        print(f"No local orbit file found. Fetching from CelesTrak for CATNR={catnr}...")
        df = get_orbit_data(catnr)
        df.to_csv(filepath, index=False)
        print(f"Saved to {filepath}")

    return {
        'full':         df,
        'raan':         df[['Date', 'RAAN']],
        'inclination':  df[['Date', 'Inclination']],
        'arg_perigee':  df[['Date', 'Arg of Perigee']],
        'sma':          df[['Date', 'SMA']],
        'eccentricity': df[['Date', 'Eccentricity']],
    }


# ── 2. Kepler Propagator ──────────────────────────────────────────────────────

def kepler_position(sma_km, ecc, inc_deg, raan_deg, argp_deg, mean_anom_deg) -> np.ndarray:
    """Compute ECI position vector from Keplerian elements."""
    inc  = np.radians(inc_deg)
    raan = np.radians(raan_deg)
    argp = np.radians(argp_deg)
    M    = np.radians(mean_anom_deg)

    # Solve Kepler's equation for eccentric anomaly
    E = M
    for _ in range(100):
        E = M + ecc * np.sin(E)

    nu = 2 * np.arctan2(np.sqrt(1 + ecc) * np.sin(E / 2),
                        np.sqrt(1 - ecc) * np.cos(E / 2))
    r = sma_km * (1 - ecc * np.cos(E))

    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)

    cos_raan, sin_raan = np.cos(raan), np.sin(raan)
    cos_argp, sin_argp = np.cos(argp), np.sin(argp)
    cos_inc,  sin_inc  = np.cos(inc),  np.sin(inc)

    x = (cos_raan * cos_argp - sin_raan * sin_argp * cos_inc) * x_orb + \
        (-cos_raan * sin_argp - sin_raan * cos_argp * cos_inc) * y_orb
    y = (sin_raan * cos_argp + cos_raan * sin_argp * cos_inc) * x_orb + \
        (-sin_raan * sin_argp + cos_raan * cos_argp * cos_inc) * y_orb
    z = (sin_argp * sin_inc) * x_orb + (cos_argp * sin_inc) * y_orb

    return np.array([x, y, z])


def propagate_kepler(row: pd.Series, dt_sec: float) -> np.ndarray:
    """Propagate a single orbital elements row forward by dt_sec seconds."""
    mu      = 398600.4418  # km^3/s^2
    n_rad_s = np.sqrt(mu / row['SMA'] ** 3)
    M_prop  = np.degrees(np.radians(row['Mean Anomaly']) + n_rad_s * dt_sec) % 360

    return kepler_position(row['SMA'], row['Eccentricity'], row['Inclination'],
                           row['RAAN'], row['Arg of Perigee'], M_prop)


# ── 3. Space Weather ──────────────────────────────────────────────────────────

def _get_kp_for_epoch(epoch: pd.Timestamp, kp_df: pd.DataFrame) -> float:
    """
    Returns the Kp value for the 3-hour interval containing the given epoch.
    Each Kp row covers [datetime, datetime + 3hr). Falls back to nearest interval.
    """
    mask  = (kp_df['datetime'] <= epoch) & (kp_df['datetime'] + pd.Timedelta(hours=3) > epoch)
    match = kp_df[mask]

    if not match.empty:
        return match.iloc[0]['kp']

    # Fallback: nearest interval if epoch is outside Kp data range
    idx = (kp_df['datetime'] - epoch).abs().argmin()
    return kp_df.iloc[idx]['kp']


def _parse_kp_text(text: str, file_type: str = 'nowcast') -> pd.DataFrame:
    """
    Parses raw GFZ Potsdam Kp text into a DataFrame.

    Two supported formats:
      'nowcast' — Kp_ap_nowcast.txt: one row per 3hr interval
                  Columns: YYYY MM DD hh.h hh._m days days_m Kp ap D
                  Kp is at index 7

      'daily'   — Kp_ap_Ap_SN_F107_since_1932.txt: one row per day
                  Columns: YYYY MM DD days days_m Bsr dB Kp1..Kp8 ap1..ap8 ...
                  Kp values are at indices 7-14
    """
    records = []
    for line in text.splitlines():
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split()
        try:
            year  = int(parts[0])
            month = int(parts[1])
            day   = int(parts[2])

            if file_type == 'nowcast':
                # One row per 3hr interval — col 3 = start hour, col 7 = Kp
                hour = float(parts[3])
                kp   = float(parts[7])
                if kp < 0:
                    continue  # missing value sentinel
                records.append({
                    'datetime': pd.Timestamp(year, month, day, int(hour), int((hour % 1) * 60)),
                    'kp': kp
                })

            elif file_type == 'daily':
                # One row per day — cols 7-14 = Kp1..Kp8 (one per 3hr interval)
                for hour_idx in range(8):
                    kp = float(parts[7 + hour_idx])
                    if kp < 0:
                        continue  # missing value sentinel
                    records.append({
                        'datetime': pd.Timestamp(year, month, day, hour_idx * 3),
                        'kp': kp
                    })

        except (ValueError, IndexError):
            continue

    return pd.DataFrame(records)


def _fetch_kp_from_source(since_year: int = None) -> pd.DataFrame:
    """
    Downloads Kp index from GFZ Potsdam and returns parsed DataFrame.

    Args:
        since_year: If provided and >= 2024, fetches the compact nowcast file.
                    If None, fetches the full historical file from 1932.
    """
    if since_year is not None and since_year >= 2024:
        url       = "https://kp.gfz-potsdam.de/app/files/Kp_ap_nowcast.txt"
        file_type = 'nowcast'
        print("  Downloading Kp nowcast file...")
    else:
        url       = "https://kp.gfz-potsdam.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt"
        file_type = 'daily'
        print("  Downloading full Kp historical file (this may take a moment)...")

    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    return _parse_kp_text(response.text, file_type=file_type), response.text


def get_kp_index(start_date: str, end_date: str, data_dir: str = "./data") -> pd.DataFrame:
    """
    Returns Kp geomagnetic index for a date range, using local cache if fresh.
    Saves both the raw .txt and parsed .csv locally.
    On first run, downloads the full historical file.
    On subsequent runs, only downloads the nowcast file if stale.

    Args:
        start_date: 'YYYY-MM-DD'
        end_date:   'YYYY-MM-DD'
        data_dir:   Folder to cache Kp files

    Returns:
        DataFrame with columns ['datetime', 'kp'] filtered to date range
    """
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "kp_index.csv")
    txt_path = os.path.join(data_dir, "kp_index.txt")

    if os.path.exists(csv_path):
        kp_df    = pd.read_csv(csv_path, parse_dates=['datetime'])
        latest   = kp_df['datetime'].max()
        age_days = (datetime.now() - latest.to_pydatetime().replace(tzinfo=None)).days

        # Sanity check — Kp must be in range 0–9; corrupt file triggers full re-download
        if kp_df['kp'].max() > 9 or kp_df['kp'].min() < 0:
            print(f"WARNING: Local Kp file has invalid values "
                  f"(min={kp_df['kp'].min():.1f}, max={kp_df['kp'].max():.1f}). Re-downloading...")
            os.remove(csv_path)
            if os.path.exists(txt_path):
                os.remove(txt_path)
            kp_df, raw_txt = _fetch_kp_from_source(since_year=None)
            with open(txt_path, 'w') as f:
                f.write(raw_txt)
            kp_df.to_csv(csv_path, index=False)
            print(f"Saved {txt_path} and {csv_path} ({len(kp_df):,} records, up to {kp_df['datetime'].max()})")

        elif age_days <= 1:
            print(f"Loaded local Kp file (latest: {latest})")
        else:
            print(f"Local Kp file is {age_days} days old. Fetching recent data from GFZ Potsdam...")
            new_df, new_txt = _fetch_kp_from_source(since_year=latest.year)

            # Append raw txt
            with open(txt_path, 'a') as f:
                f.write(new_txt)

            # Merge parsed data: trim overlap, append new, deduplicate
            kp_df = pd.concat([
                kp_df[kp_df['datetime'] < new_df['datetime'].min()],
                new_df
            ]).drop_duplicates('datetime').sort_values('datetime').reset_index(drop=True)

            kp_df.to_csv(csv_path, index=False)
            print(f"Updated {csv_path} (now covers up to {kp_df['datetime'].max()})")
    else:
        print("No local Kp file found. Downloading full historical file from GFZ Potsdam...")
        kp_df, raw_txt = _fetch_kp_from_source(since_year=None)

        # Save both raw txt and parsed csv
        with open(txt_path, 'w') as f:
            f.write(raw_txt)
        kp_df.to_csv(csv_path, index=False)
        print(f"Saved {txt_path} and {csv_path} ({len(kp_df):,} records, up to {kp_df['datetime'].max()})")

    return kp_df[(kp_df['datetime'] >= start_date) &
                 (kp_df['datetime'] <= end_date)].reset_index(drop=True)


# ── 4. Manoeuvre Detection ────────────────────────────────────────────────────

def analyse_maneuvers_kepler(catnr: int,
                             data_dir: str = "./data",
                             output_dir: str = "./output",
                             max_age_days: int = 1,
                             delta_sma_threshold: float = 1.0,
                             delta_ecc_threshold: float = 5e-5,
                             kp_threshold: float = 5.0) -> pd.DataFrame:
    """
    Fetches orbital elements, detects manoeuvres by comparing consecutive epochs,
    and cross-checks against space weather (Kp index) to confirm manoeuvres.

    Args:
        catnr:                NORAD catalog number
        data_dir:             Folder for local data cache (TLE, Kp CSV files)
        output_dir:           Folder for output plots and results
        max_age_days:         Cache freshness threshold (days)
        delta_sma_threshold:  SMA change threshold to flag a manoeuvre (km)
        delta_ecc_threshold:  Eccentricity change threshold to flag a manoeuvre
        kp_threshold:         Kp value above which space weather is considered disturbed

    Returns:
        DataFrame with epoch pairs, deltas, Kp values, and manoeuvre flags
    """
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    elements = load_orbit_data(catnr, data_dir=data_dir, max_age_days=max_age_days)
    df = elements['full'].copy().sort_values('Date').reset_index(drop=True)

    if 'Mean Anomaly' not in df.columns:
        print("'Mean Anomaly' not in data — approximating as 0.0 (epoch is perigee pass)")
        df['Mean Anomaly'] = 0.0

    missing = [c for c in ['Date', 'SMA', 'Eccentricity', 'Inclination', 'RAAN', 'Arg of Perigee']
               if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in orbit data: {missing}")

    # ── Smooth orbital elements with rolling median to reduce TLE noise ────────
    for col in ['SMA', 'Eccentricity', 'Inclination', 'RAAN', 'Arg of Perigee']:
        df[f'{col}_smooth'] = df[col].rolling(3, center=True, min_periods=1).median()

    # ── Compute epoch pair deltas ─────────────────────────────────────────────
    results = []
    for i in range(len(df) - 1):
        row_now  = df.iloc[i]
        row_next = df.iloc[i + 1]

        gap_sec  = (row_next['Date'] - row_now['Date']).total_seconds()
        gap_days = gap_sec / 86400

        if gap_days > 10:  # skip data gaps
            continue

        pos_predicted = propagate_kepler(row_now, gap_sec)
        pos_actual    = kepler_position(row_next['SMA'], row_next['Eccentricity'],
                                        row_next['Inclination'], row_next['RAAN'],
                                        row_next['Arg of Perigee'], row_next['Mean Anomaly'])

        # Use smoothed values for delta computation
        delta_sma = row_next['SMA_smooth'] - row_now['SMA_smooth']
        delta_ecc = row_next['Eccentricity_smooth'] - row_now['Eccentricity_smooth']
        delta_inc = row_next['Inclination_smooth'] - row_now['Inclination_smooth']

        # ── Background drift rate over ±3 days ────────────────────────────────
        window = df[(df['Date'] >= row_now['Date'] - pd.Timedelta(days=3)) &
                    (df['Date'] <= row_now['Date'] + pd.Timedelta(days=3))]
        if len(window) >= 3:
            drift_rate_sma = np.polyfit(
                (window['Date'] - row_now['Date']).dt.total_seconds() / 86400,
                window['SMA_smooth'], deg=1
            )[0]  # km/day
        else:
            drift_rate_sma = 0.0

        # Excess delta = observed change minus expected background drift
        expected_sma_drift = drift_rate_sma * gap_days
        excess_delta_sma   = delta_sma - expected_sma_drift

        results.append({
            'epoch_from':       row_now['Date'],
            'epoch_to':         row_next['Date'],
            'gap_days':         round(gap_days, 3),
            'error_km':         round(np.linalg.norm(pos_predicted - pos_actual), 3),
            'delta_sma':        round(delta_sma, 3),
            'delta_ecc':        round(delta_ecc, 6),
            'delta_inc':        round(delta_inc, 4),
            'excess_delta_sma': round(excess_delta_sma, 3),
            'drift_rate_sma':   round(drift_rate_sma, 4),
        })

    result_df = pd.DataFrame(results)

    # ── Space weather Kp lookup ───────────────────────────────────────────────
    start = result_df['epoch_from'].min().strftime('%Y-%m-%d')
    end   = result_df['epoch_to'].max().strftime('%Y-%m-%d')
    kp_df = get_kp_index(start, end, data_dir=data_dir)

    result_df['kp'] = result_df['epoch_from'].apply(
        lambda ep: _get_kp_for_epoch(ep, kp_df)
    )

    # ── Manoeuvre classification ──────────────────────────────────────────────
    result_df['bad_space_weather'] = result_df['kp'] >= kp_threshold
    result_df['likely_maneuver']   = (
        (result_df['excess_delta_sma'].abs() > delta_sma_threshold) |
        (result_df['delta_ecc'].abs()         > delta_ecc_threshold)
    )
    result_df['confirmed_maneuver'] = result_df['likely_maneuver'] & ~result_df['bad_space_weather']
    result_df['uncertain_maneuver'] = result_df['likely_maneuver'] &  result_df['bad_space_weather']

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n── Kepler Manoeuvre Analysis: CATNR {catnr} ─────────────────────")
    print(f"  Epoch pairs analysed:      {len(result_df)}")
    print(f"  delta_sma threshold:       {delta_sma_threshold} km")
    print(f"  delta_ecc threshold:       {delta_ecc_threshold}")
    print(f"  Kp threshold:              {kp_threshold}")
    print(f"  Likely manoeuvres:         {result_df['likely_maneuver'].sum()}")
    print(f"  Confirmed manoeuvres:      {result_df['confirmed_maneuver'].sum()}")
    print(f"  Uncertain (bad weather):   {result_df['uncertain_maneuver'].sum()}")

    def _print_maneuver_rows(rows: pd.DataFrame, label: str) -> None:
        for _, r in rows.iterrows():
            print(f"  {r['epoch_from']}  →  {r['epoch_to']}  "
                  f"(gap: {r['gap_days']:.3f} days)  "
                  f"ΔSMA: {r['delta_sma']:+.3f} km  "
                  f"excess ΔSMA: {r['excess_delta_sma']:+.3f} km  "
                  f"drift: {r['drift_rate_sma']:+.4f} km/day  "
                  f"ΔEcc: {r['delta_ecc']:+.2e}  "
                  f"ΔInc: {r['delta_inc']:+.4f}°  "
                  f"Kp: {r['kp']:.1f}  [{label}]")

    if result_df['confirmed_maneuver'].any():
        print(f"\n── Confirmed Manoeuvre Epochs ───────────────────────────────────")
        _print_maneuver_rows(result_df[result_df['confirmed_maneuver']], 'CONFIRMED')

    if result_df['uncertain_maneuver'].any():
        print(f"\n── Uncertain Epochs (possible manoeuvre, bad space weather) ─────")
        _print_maneuver_rows(result_df[result_df['uncertain_maneuver']], 'UNCERTAIN')

    # ── Plot ──────────────────────────────────────────────────────────────────
    _plot_results(result_df, catnr,
                  delta_sma_threshold=delta_sma_threshold,
                  delta_ecc_threshold=delta_ecc_threshold,
                  kp_threshold=kp_threshold,
                  output_dir=output_dir)

    return result_df


def _plot_results(result_df: pd.DataFrame, catnr: int,
                  delta_sma_threshold: float,
                  delta_ecc_threshold: float,
                  kp_threshold: float,
                  output_dir: str = "./output") -> None:
    """Plot Keplerian element deltas and Kp with manoeuvre markers."""
    os.makedirs(output_dir, exist_ok=True)

    confirmed = result_df[result_df['confirmed_maneuver']]
    uncertain = result_df[result_df['uncertain_maneuver']]

    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    fig.suptitle(f'Kepler Manoeuvre Detection — CATNR {catnr}', fontsize=13, y=0.98)

    def _scatter_flags(ax, col):
        if not confirmed.empty:
            ax.scatter(confirmed['epoch_to'], confirmed[col],
                       color='red', zorder=5, s=30, label='Confirmed')
        if not uncertain.empty:
            ax.scatter(uncertain['epoch_to'], uncertain[col],
                       color='orange', zorder=5, marker='^', s=30, label='Uncertain')

    # Panel 1: ΔSMA (raw and excess over background drift)
    axes[0].plot(result_df['epoch_to'], result_df['delta_sma'],
                 marker='o', markersize=3, color='darkorange',
                 label='ΔSMA (smoothed)', alpha=0.6, linewidth=0.8)
    axes[0].plot(result_df['epoch_to'], result_df['excess_delta_sma'],
                 marker='o', markersize=3, color='saddlebrown',
                 label='Excess ΔSMA (above drift)', linewidth=0.8)
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[0].axhline( delta_sma_threshold, color='red', linestyle=':', linewidth=1,
                     label=f'±{delta_sma_threshold} km')
    axes[0].axhline(-delta_sma_threshold, color='red', linestyle=':', linewidth=1)
    _scatter_flags(axes[0], 'excess_delta_sma')
    axes[0].set_ylabel('ΔSMA (km)')
    axes[0].legend(fontsize=8, loc='upper right')

    # Panel 2: ΔEccentricity
    axes[1].plot(result_df['epoch_to'], result_df['delta_ecc'],
                 marker='o', markersize=3, color='green', linewidth=0.8)
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[1].axhline( delta_ecc_threshold, color='red', linestyle=':', linewidth=1,
                     label=f'±{delta_ecc_threshold}')
    axes[1].axhline(-delta_ecc_threshold, color='red', linestyle=':', linewidth=1)
    _scatter_flags(axes[1], 'delta_ecc')
    axes[1].set_ylabel('ΔEccentricity')
    axes[1].legend(fontsize=8, loc='upper right')

    # Panel 3: Position error
    axes[2].plot(result_df['epoch_to'], result_df['error_km'],
                 marker='o', markersize=3, color='steelblue', linewidth=0.8)
    _scatter_flags(axes[2], 'error_km')
    axes[2].set_ylabel('Position Error (km)')
    axes[2].legend(fontsize=8, loc='upper right')

    # Panel 4: Kp index
    axes[3].plot(result_df['epoch_to'], result_df['kp'],
                 marker='o', markersize=3, color='purple', linewidth=0.8)
    axes[3].axhline(kp_threshold, color='red', linestyle='--', linewidth=1,
                    label=f'Kp threshold ({kp_threshold})')
    axes[3].fill_between(result_df['epoch_to'], result_df['kp'], kp_threshold,
                         where=result_df['kp'] >= kp_threshold,
                         color='red', alpha=0.2, label='Disturbed')
    axes[3].set_ylabel('Kp')
    axes[3].set_xlabel('Epoch')
    axes[3].legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'kepler_maneuver_detection_{catnr}.png')
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"Plot saved to {plot_path}")


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os as _os

    CATNR = int(_os.environ.get("CATNR", "60547"))

    df_results = analyse_maneuvers_kepler(
        catnr               = CATNR,
        data_dir            = "./data",
        output_dir          = "./output",
        delta_sma_threshold = 0.1,
        delta_ecc_threshold = 5e-5,
        kp_threshold        = 5.0,
    )

    print(df_results[['epoch_from', 'epoch_to', 'gap_days',
                       'error_km', 'delta_sma', 'excess_delta_sma',
                       'drift_rate_sma', 'delta_ecc', 'delta_inc',
                       'kp', 'confirmed_maneuver', 'uncertain_maneuver']]
          .to_string(index=False))
