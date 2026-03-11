# Satellite Manoeuvre Detection

Detects satellite manoeuvres from public TLE history using two independent methods: a fast Keplerian element delta approach and a higher-fidelity Orbit Determination (OD) propagator. Results are cross-checked against the Kp geomagnetic index to distinguish genuine manoeuvres from space-weather artefacts.

---

## Methods

### Kepler (`orbit_analysis_kepler.py`)
Fetches orbital element history from [CelesTrak](https://celestrak.org). For each consecutive epoch pair, it propagates the current Keplerian elements forward and compares the predicted position against the next epoch's reported elements. Manoeuvres are flagged when the SMA or eccentricity step-change exceeds a threshold after subtracting the expected background drift rate.

- **No account required** — data sourced from CelesTrak's public API
- Fast to run — no numerical integration
- Best for detecting clear altitude-raising or plane-change manoeuvres

### OD (`orbit_analysis_od.py`)
Downloads full TLE history from [Space-Track](https://www.space-track.org) (free account required). For each TLE pair, it runs SGP4 at the epoch to get an osculating ECI state, then propagates forward using RK4 numerical integration with J2 oblateness and atmospheric drag (US Standard Atmosphere 1976, co-rotating with Earth). The predicted state is compared against the next TLE's SGP4 state. Manoeuvres are flagged via an adaptive threshold derived from the data's own noise floor (median + N×MAD of gap-normalised velocity residuals), with a secondary B* step-change signal.

- Requires a free [Space-Track account](https://www.space-track.org/auth/createAccount)
- Higher sensitivity — detects small manoeuvres missed by element deltas
- Uses `python-sgp4` if installed; falls back to a built-in SGP4 implementation

Both methods use the **Kp geomagnetic index** from [GFZ Potsdam](https://kp.gfz-potsdam.de) to classify detections as *confirmed* (low Kp) or *uncertain* (disturbed space weather).

---

## File Structure

```
.
├── main.py                    # Entry point — runs either or both methods
├── orbit_analysis_kepler.py   # Kepler method
├── orbit_analysis_od.py       # OD method
├── data/                      # Auto-created: cached TLE and Kp files
│   ├── orbit_<catnr>.csv      # CelesTrak orbital elements (Kepler)
│   ├── tle_history_<catnr>.txt# Space-Track TLE history (OD)
│   ├── kp_index.csv           # Parsed Kp index cache
│   └── kp_index.txt           # Raw Kp index download
└── output/                    # Auto-created: plots
    ├── kepler_maneuver_detection_<catnr>.png
    └── od_maneuver_detection_<catnr>.png
```

---

## Installation

```bash
pip install numpy pandas matplotlib requests

# Optional but recommended — enables full SGP4/SDP4 (higher accuracy)
pip install sgp4
```

Python 3.10+ is required (uses `list[dict]` and `dict[str, ...]` type hints).

---

## Quick Start

```bash
# Run both methods with interactive credential prompt
python main.py

# Run Kepler only (no account needed)
python main.py --method kepler

# Run OD only
python main.py --method od

# Pass credentials via environment variables (recommended for scripts)
CATNR=60547 SPACETRACK_USER=you@example.com SPACETRACK_PASS=secret python main.py
```

---

## Usage

### `main.py` — CLI options

| Argument | Default | Description |
|---|---|---|
| `--method` | `both` | `kepler`, `od`, or `both` |
| `--catnr` | `60547` | NORAD catalog number |
| `--data-dir` | `./data` | Folder for cached data files |
| `--output-dir` | `./output` | Folder for output plots |
| `--kp-threshold` | `5.0` | Kp above which weather is "disturbed" |
| `--delta-sma` | `0.1` | Kepler: SMA change threshold (km) |
| `--delta-ecc` | `5e-5` | Kepler: eccentricity change threshold |
| `--sigma` | `5.0` | OD: adaptive threshold multiplier (N×MAD) |
| `--max-gap` | `10.0` | OD: skip TLE pairs with gap > N days |
| `--no-bstar` | — | OD: disable B*-calibrated drag |

### Environment variables

| Variable | Description |
|---|---|
| `CATNR` | NORAD catalog number (overridden by `--catnr`) |
| `SPACETRACK_USER` | Space-Track login email |
| `SPACETRACK_PASS` | Space-Track password |

### Using as a library

```python
from orbit_analysis_kepler import analyse_maneuvers_kepler
from orbit_analysis_od import analyse_maneuvers_od

# Kepler method
df_kep = analyse_maneuvers_kepler(
    catnr               = 60547,
    data_dir            = "./data",
    output_dir          = "./output",
    delta_sma_threshold = 0.1,      # km
    delta_ecc_threshold = 5e-5,
    kp_threshold        = 5.0,
)

# OD method
df_od = analyse_maneuvers_od(
    catnr                = 60547,
    username             = "you@example.com",
    password             = "your_password",
    data_dir             = "./data",
    output_dir           = "./output",
    sigma_multiplier     = 5.0,
    kp_threshold         = 5.0,
    max_gap_days         = 10.0,
    use_bstar            = True,
    bstar_noise_max      = 1e-3,
    delta_bstar_threshold= 5e-5,
)
```

Both functions return a `pandas.DataFrame` — see [Output columns](#output-columns) below.

---

## Output Columns

### Kepler DataFrame

| Column | Description |
|---|---|
| `epoch_from` / `epoch_to` | Epoch pair timestamps |
| `gap_days` | Time between epochs (days) |
| `error_km` | Propagation position error (km) |
| `delta_sma` | Smoothed SMA change (km) |
| `excess_delta_sma` | SMA change minus background drift (km) |
| `drift_rate_sma` | Background SMA drift rate (km/day) |
| `delta_ecc` | Smoothed eccentricity change |
| `delta_inc` | Smoothed inclination change (degrees) |
| `kp` | Kp index at epoch |
| `likely_maneuver` | True if SMA or ecc threshold exceeded |
| `confirmed_maneuver` | `likely_maneuver` AND low Kp |
| `uncertain_maneuver` | `likely_maneuver` AND high Kp |

### OD DataFrame

| Column | Description |
|---|---|
| `epoch_from` / `epoch_to` | Epoch pair timestamps |
| `gap_days` | Time between epochs (days) |
| `vel_residual_ms` | Velocity residual magnitude (m/s) |
| `pos_residual_km` | Position residual magnitude (km) |
| `vel_res_per_day` | Gap-normalised velocity residual (m/s/day) |
| `sma` / `delta_sma` | Semi-major axis and change (km) |
| `delta_ecc` / `delta_inc` | Eccentricity and inclination changes |
| `bstar` / `delta_bstar` | B* drag term and step-change |
| `od_flag` | True if velocity residual exceeds adaptive threshold |
| `bstar_flag` | True if B* step-change exceeds threshold |
| `vel_res_inf` | True if residual was non-finite (degenerate propagation) |
| `kp` | Kp index at epoch |
| `likely_maneuver` | `od_flag` OR `bstar_flag` |
| `confirmed_maneuver` | `likely_maneuver` AND low Kp |
| `uncertain_maneuver` | `likely_maneuver` AND high Kp |

---

## Data Caching

Both modules cache downloaded data locally and refresh incrementally:

- **TLE history** (`orbit_analysis_od`): re-uses the local file if the latest TLE epoch is less than 1 day old; otherwise fetches only TLEs newer than the latest cached epoch and appends them.
- **CelesTrak elements** (`orbit_analysis_kepler`): re-uses the local CSV if the latest date is within `max_age_days` (default 1).
- **Kp index**: shared between both modules. On first run, downloads the full GFZ Potsdam historical file (1932–present). On subsequent runs, only fetches the compact nowcast file to append recent data.

Delete files in `./data/` to force a full re-download.

---

## Notes

- NORAD catalog numbers can be looked up on [CelesTrak](https://celestrak.org) or [Space-Track](https://www.space-track.org).
- The OD method uses `python-sgp4` (Vallado 2006 / Brandon Rhodes) when installed, which provides full SGP4/SDP4 with automatic near-Earth/deep-space selection. The built-in fallback covers near-Earth orbits only (period < 225 min).
- The adaptive OD threshold is computed from the dataset itself (median + N×MAD of gap-normalised residuals), so it self-calibrates to each satellite's noise floor. Increase `--sigma` to reduce false positives on noisy TLE histories.
- Manoeuvres detected during geomagnetically disturbed periods (Kp ≥ threshold) are marked `uncertain` rather than `confirmed` because atmospheric drag enhancements can mimic manoeuvre signatures in both methods.
