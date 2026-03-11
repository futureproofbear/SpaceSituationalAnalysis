# main.py

A Python script for detecting satellite manoeuvres from historical orbital element data, using CelesTrak as the data source and GFZ Potsdam's Kp index for space weather cross-checking.

---

## Overview

The script fetches orbital element history for a satellite by NORAD catalog number, computes epoch-to-epoch changes in Semi-Major Axis (SMA) and eccentricity, and flags likely manoeuvres. Detections are cross-checked against geomagnetic activity (Kp index) to distinguish confirmed manoeuvres from uncertain ones during disturbed space weather.

All downloaded data is cached locally under `./data/` to avoid redundant downloads.

---

## Requirements

```
pip install requests numpy pandas matplotlib
```

---

## Quick Start

```python
from orbit_analysis import analyse_maneuvers

df_results = analyse_maneuvers(66748)
```

This will:
1. Download orbital element history for NORAD XXXXX from CelesTrak
2. Download the Kp geomagnetic index from GFZ Potsdam
3. Detect manoeuvres and cross-check against space weather
4. Print a summary and save a plot to `maneuver_detection_XXXXX.png`

---

## Functions

### `get_orbit_data(catnr)`
Fetches orbital element history directly from CelesTrak's graph data page.

| Parameter | Type | Description |
|-----------|------|-------------|
| `catnr` | `int` | NORAD catalog number |

Returns a `DataFrame` with columns: `Date`, `RAAN`, `Inclination`, `Arg of Perigee`, `SMA`, `Eccentricity`.

---

### `load_orbit_data(catnr, data_dir, max_age_days)`
Loads orbital data from local cache if fresh, otherwise fetches from CelesTrak. Saves to `./data/orbit_{catnr}.csv`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `catnr` | `int` | — | NORAD catalog number |
| `data_dir` | `str` | `"./data"` | Folder to cache CSV files |
| `max_age_days` | `int` | `1` | Re-download if local file is older than this |

Returns a `dict` of DataFrames keyed by element name: `full`, `raan`, `inclination`, `arg_perigee`, `sma`, `eccentricity`.

---

### `analyse_maneuvers(catnr, data_dir, max_age_days, delta_sma_threshold, delta_ecc_threshold, kp_threshold)`
Main function. Loads orbital data, computes smoothed epoch-to-epoch deltas, subtracts background drift, looks up Kp for each epoch, and classifies manoeuvres.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `catnr` | `int` | — | NORAD catalog number |
| `data_dir` | `str` | `"./data"` | Folder for local cache |
| `max_age_days` | `int` | `1` | Cache freshness threshold |
| `delta_sma_threshold` | `float` | `1.0` | Excess SMA change to flag a manoeuvre (km) |
| `delta_ecc_threshold` | `float` | `5e-5` | Eccentricity change to flag a manoeuvre |
| `kp_threshold` | `float` | `5.0` | Kp value above which space weather is disturbed |

Returns a `DataFrame` with one row per epoch pair and the following columns:

| Column | Description |
|--------|-------------|
| `epoch_from` | Start of the epoch interval |
| `epoch_to` | End of the epoch interval |
| `gap_days` | Time between epochs (days) |
| `error_km` | Kepler propagation error (km) |
| `delta_sma` | Smoothed SMA change (km) |
| `delta_ecc` | Smoothed eccentricity change |
| `delta_inc` | Smoothed inclination change (degrees) |
| `excess_delta_sma` | SMA change above background drift (km) |
| `drift_rate_sma` | Background SMA drift rate (km/day) from ±3-day linear fit |
| `kp` | Kp value for the 3-hour interval containing `epoch_from` |
| `bad_space_weather` | `True` if Kp ≥ kp_threshold |
| `likely_maneuver` | `True` if excess_delta_sma or delta_ecc exceeds threshold |
| `confirmed_maneuver` | `True` if likely_maneuver AND NOT bad_space_weather |
| `uncertain_maneuver` | `True` if likely_maneuver AND bad_space_weather |

---

### `get_kp_index(start_date, end_date, data_dir)`
Returns Kp geomagnetic index for a date range with local caching.

- **First run**: downloads the full historical file from GFZ Potsdam (since 1932, ~5 MB). Saves `kp_index.csv` and `kp_index.txt` to `data_dir`.
- **Subsequent runs**: loads from local cache if less than 1 day old. If stale, downloads only the compact nowcast file and appends it to the local cache.
- **Corrupt file detection**: if Kp values fall outside 0–9, the cache is automatically deleted and re-downloaded.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start_date` | `str` | — | Start date `'YYYY-MM-DD'` |
| `end_date` | `str` | — | End date `'YYYY-MM-DD'` |
| `data_dir` | `str` | `"./data"` | Folder to cache Kp files |

---

## Detection Logic

### 1. Rolling Median Smoothing
All orbital elements are smoothed with a 3-epoch centred rolling median before delta computation. This reduces noise from individual noisy TLE fits.

### 2. Background Drift Subtraction
For each epoch pair, a linear trend is fitted to the smoothed SMA over a ±3-day window to estimate the natural atmospheric drag decay rate (km/day). The `excess_delta_sma` is the observed SMA change minus the expected background drift. This separates deliberate orbit raises from passive decay.

### 3. Space Weather Filter
Each epoch is matched to its exact 3-hour Kp interval (not a daily max). If Kp ≥ `kp_threshold` at the time of a detected change, the event is flagged as `uncertain_maneuver` rather than `confirmed_maneuver` — the orbital change may be real but could also be an artefact of disturbed atmospheric conditions.

### Manoeuvre Classification

| Condition | Classification |
|-----------|---------------|
| `excess_delta_sma > threshold` OR `delta_ecc > threshold`, AND Kp < threshold | `confirmed_maneuver` |
| `excess_delta_sma > threshold` OR `delta_ecc > threshold`, AND Kp ≥ threshold | `uncertain_maneuver` |
| Neither delta exceeds threshold | No manoeuvre |

---

## Output Plot

A 4-panel plot is saved to `maneuver_detection_{catnr}.png`:

- **Panel 1 — ΔSMA**: raw smoothed delta (faded) and excess delta above drift (solid). Red circles = confirmed, orange triangles = uncertain.
- **Panel 2 — ΔEccentricity**: epoch-to-epoch eccentricity changes with threshold lines.
- **Panel 3 — Position Error**: Kepler propagation error in km.
- **Panel 4 — Kp Index**: geomagnetic activity with threshold line and shaded storm periods.

---

## Local File Structure

```
./data/
├── orbit_{catnr}.csv      # Orbital element history from CelesTrak
├── kp_index.csv           # Parsed Kp index (full history)
└── kp_index.txt           # Raw Kp text file from GFZ Potsdam
```

---

## Data Sources

| Data | Source | URL |
|------|--------|-----|
| Orbital elements | CelesTrak | `celestrak.org/NORAD/elements/graph-orbit-data.php` |
| Kp index (historical) | GFZ Potsdam | `kp.gfz-potsdam.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt` |
| Kp index (nowcast) | GFZ Potsdam | `kp.gfz-potsdam.de/app/files/Kp_ap_nowcast.txt` |

---

## Kp Index Reference

| Kp | Activity | Effect |
|----|----------|--------|
| 0–1 | Quiet | Nominal drag |
| 2–3 | Unsettled | Minor fluctuations |
| 4 | Active | Noticeable drag increase |
| 5 | Minor storm (G1) | Thermosphere expanding |
| 6 | Moderate storm (G2) | Significant drag increase |
| 7 | Strong storm (G3) | Major density surge |
| 8–9 | Severe/Extreme (G4–G5) | Extreme drag |
