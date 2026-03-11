"""
orbit_analysis_od.py
--------------------
Manoeuvre detection using Orbit Determination (OD) from Space-Track TLE history.

Data source : https://www.space-track.org  (free account required)
Method      : Propagate each TLE epoch forward with J2 + drag (RK4) to the next
              epoch, compare predicted vs actual state vector, flag large velocity
              residuals as manoeuvres, cross-check with Kp space weather index.

Usage:
    python orbit_analysis_od.py
    CATNR=60547 SPACETRACK_USER=you@example.com SPACETRACK_PASS=secret python orbit_analysis_od.py
    or import and call analyse_maneuvers_od(catnr, username, password)
"""

import os
import re
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from datetime import datetime

# Try to import python-sgp4 (pip install sgp4).
# If unavailable, fall back to the hand-rolled SGP4 implementation below.
try:
    from sgp4.api import Satrec, jday as sgp4_jday
    _SGP4_LIB = True
except ImportError:
    _SGP4_LIB = False


# ── Constants ─────────────────────────────────────────────────────────────────

MU      = 398600.4418   # km^3/s^2
J2      = 1.08263e-3
RE      = 6378.137      # km
OMEGA_E = 7.2921150e-5  # Earth rotation rate rad/s

# US Standard Atmosphere 1976 — piecewise exponential layers.
# Each entry: (base_alt_km, base_density_kg/m^3, scale_height_km)
# Covers 0–1000 km; above 1000 km density is negligible for drag.
_ATMO_LAYERS = np.array([
    (0,    1.225,      8.44),
    (25,   3.899e-2,   6.49),
    (30,   1.774e-2,   6.75),
    (40,   3.972e-3,   7.47),
    (50,   1.057e-3,   8.38),
    (60,   3.206e-4,   7.71),
    (70,   8.770e-5,   6.34),
    (80,   1.905e-5,   5.38),
    (90,   3.396e-6,   5.74),
    (100,  5.604e-7,   6.15),
    (110,  9.708e-8,   8.10),
    (120,  2.222e-8,   11.4),
    (130,  8.152e-9,   17.0),
    (140,  3.831e-9,   24.0),
    (150,  2.076e-9,   32.0),
    (180,  5.194e-10,  51.0),
    (200,  2.541e-10,  57.0),
    (250,  6.073e-11,  75.5),
    (300,  1.916e-11,  89.4),
    (350,  7.014e-12,  105.8),
    (400,  2.803e-12,  124.6),
    (450,  1.184e-12,  148.1),
    (500,  5.215e-13,  176.0),
    (600,  1.137e-13,  245.4),
    (700,  3.070e-14,  338.0),
    (800,  1.136e-14,  408.0),
    (900,  5.759e-15,  516.0),
    (1000, 3.561e-15,  830.0),
])


# ── SGP4 backend selection ────────────────────────────────────────────────────
if _SGP4_LIB:
    print("SGP4 backend: python-sgp4 (full SGP4/SDP4, Vallado 2006)")
else:
    print("SGP4 backend: hand-rolled fallback (SGP4 near-Earth only). "
          "Install python-sgp4 for full accuracy: pip install sgp4")


# ── 1. Space-Track TLE Download ───────────────────────────────────────────────

def _spacetrack_session(username: str, password: str) -> requests.Session:
    """Authenticate with Space-Track and return an active session."""
    session = requests.Session()
    resp = session.post(
        "https://www.space-track.org/ajaxauth/login",
        data={"identity": username, "password": password}
    )
    resp.raise_for_status()
    if "Failed" in resp.text:
        raise ValueError("Space-Track login failed — check username/password.")
    return session


def fetch_tle_history(catnr: int, username: str, password: str,
                      data_dir: str = "./data") -> list[dict]:
    """
    Downloads TLE history for a satellite from Space-Track.
    Caches to ./data/tle_history_{catnr}.txt.

    Freshness is judged by the epoch of the most recent TLE in the local file,
    not the file modification time. If the latest TLE epoch is less than 1 day
    old, the local file is used as-is. Otherwise only TLEs newer than the
    latest local epoch are fetched and appended.

    Args:
        catnr:     NORAD catalog number
        username:  Space-Track username (email)
        password:  Space-Track password
        data_dir:  Folder to cache TLE files

    Returns:
        List of dicts with keys: name, line1, line2, epoch
    """
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, f"tle_history_{catnr}.txt")

    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            raw = f.read()

        existing = _parse_tle_text(raw)
        if not existing:
            print("Local TLE file is empty or unparseable. Re-downloading full history...")
            raw = _download_tle_history(catnr, username, password)
            with open(filepath, 'w') as f:
                f.write(raw)
            print(f"Saved to {filepath}")
            return _parse_tle_text(raw)

        latest_epoch = max(t['epoch'] for t in existing)
        age_days     = (datetime.now() - latest_epoch).total_seconds() / 86400

        if age_days <= 1:
            print(f"Local TLE history is current — latest epoch: "
                  f"{latest_epoch.strftime('%Y-%m-%d %H:%M')}  ({len(existing)} TLEs)")
            return existing

        # Fetch only TLEs newer than the latest local epoch
        since_str = latest_epoch.strftime('%Y-%m-%d%%20%H:%M:%S')
        print(f"Latest local TLE epoch: {latest_epoch.strftime('%Y-%m-%d %H:%M')} "
              f"({age_days:.1f} days ago). Fetching newer TLEs from Space-Track...")
        new_raw = _download_tle_history(catnr, username, password, since=latest_epoch)

        new_tles = _parse_tle_text(new_raw)
        if new_tles:
            # Append only genuinely new lines to the file
            with open(filepath, 'a') as f:
                f.write('\n' + new_raw.strip())
            print(f"Appended {len(new_tles)} new TLEs → {filepath}")
        else:
            print("No new TLEs available from Space-Track.")

        # Re-parse the full merged file
        with open(filepath, 'r') as f:
            raw = f.read()
        return _parse_tle_text(raw)

    else:
        print(f"No local TLE file found. Downloading full history from Space-Track...")
        raw = _download_tle_history(catnr, username, password)
        with open(filepath, 'w') as f:
            f.write(raw)
        tles = _parse_tle_text(raw)
        print(f"Saved {len(tles)} TLEs to {filepath}")
        return tles


def _download_tle_history(catnr: int, username: str, password: str,
                          since: datetime = None) -> str:
    """
    Downloads TLE history from Space-Track.

    Args:
        catnr:  NORAD catalog number
        since:  If provided, only fetch TLEs with epoch > this datetime.
                If None, fetch the full history.
    """
    base = "https://www.space-track.org/basicspacedata/query/class/gp_history"
    if since is not None:
        since_str = since.strftime('%Y-%m-%d%%20%H:%M:%S')
        url = (f"{base}/NORAD_CAT_ID/{catnr}"
               f"/EPOCH/%3E{since_str}"
               f"/orderby/TLE_LINE1 ASC/format/tle")
    else:
        url = (f"{base}/NORAD_CAT_ID/{catnr}"
               f"/orderby/TLE_LINE1 ASC/format/tle")

    session = _spacetrack_session(username, password)
    resp    = session.get(url)
    resp.raise_for_status()
    return resp.text


def _parse_tle_epoch(line1: str) -> datetime:
    """Parse epoch datetime from TLE line 1."""
    epoch_str = line1[18:32].strip()
    year      = int(epoch_str[:2])
    year     += 2000 if year < 57 else 1900
    day_frac  = float(epoch_str[2:])
    return datetime(year, 1, 1) + pd.Timedelta(days=day_frac - 1)


def _parse_tle_text(raw: str) -> list[dict]:
    """Parse raw 3-line TLE text into a list of dicts."""
    lines  = [l.strip() for l in raw.splitlines() if l.strip()]
    tles   = []

    i = 0
    while i < len(lines) - 1:
        # Handle both 2-line and 3-line TLE formats
        if lines[i].startswith('1 ') and i + 1 < len(lines) and lines[i+1].startswith('2 '):
            line1, line2 = lines[i], lines[i+1]
            name = f"CATNR {line1[2:7].strip()}"
            i += 2
        elif not lines[i].startswith('1 ') and not lines[i].startswith('2 ') \
             and i + 2 < len(lines) \
             and lines[i+1].startswith('1 ') and lines[i+2].startswith('2 '):
            name  = lines[i]
            line1 = lines[i+1]
            line2 = lines[i+2]
            i += 3
        else:
            i += 1
            continue

        tles.append({
            'name':  name,
            'line1': line1,
            'line2': line2,
            'epoch': _parse_tle_epoch(line1),
        })

    print(f"Parsed {len(tles)} TLEs")
    return tles


# ── 2. TLE Parsing ────────────────────────────────────────────────────────────

def _parse_bstar(s: str) -> float:
    """
    Parse B* drag term from TLE line 1 columns 54-61.
    TLE format: SMMMMM+EE where S=sign, MMMMM=mantissa*10^5, EE=exponent.
    Result is in units of 1/Earth-radius (SGP4 native units).
    """
    s = s.strip()
    try:
        if len(s) == 8:
            sign     = -1.0 if s[0] == '-' else 1.0
            mantissa = float(s[1:6]) * 1e-5
            exp      = int(s[6:8])
            return sign * mantissa * 10 ** exp
        return float(s)
    except Exception:
        return 0.0


def _tle_to_mean(line1: str, line2: str) -> dict:
    """
    Extract raw TLE mean elements (SGP4 mean motion theory, not osculating).
    These are NOT Keplerian osculating elements and must not be fed directly
    into a two-body state vector converter.

    Returns dict with mean elements and B* for use by _sgp4_state().
    """
    ecc       = float('0.' + line2[26:33])
    inc       = float(line2[8:16])
    raan      = float(line2[17:25])
    argp      = float(line2[34:42])
    mean_anom = float(line2[43:51])
    mean_mot  = float(line2[52:63])   # rev/day (SGP4 mean motion)
    bstar     = _parse_bstar(line1[53:61])

    # ndot, nddot (not used in SGP4 simplified but stored)
    ndot  = float(line1[33:43])
    return {
        'ecc':        ecc,
        'inc':        inc,
        'raan':       raan,
        'argp':       argp,
        'mean_anom':  mean_anom,
        'mean_mot':   mean_mot,   # rev/day
        'bstar':      bstar,
        'ndot':       ndot,
    }


# ── 3. SGP4 → Osculating ECI State ───────────────────────────────────────────
#
# TLE mean elements live in the SGP4 mean element theory. Converting them to
# an osculating ECI state vector requires running SGP4 at t=0 (epoch).
# This is the ONLY correct way — treating mean elements as Keplerian elements
# produces position errors of several km and velocity errors of tens of m/s.
#
# Two implementations are provided:
#   _sgp4_lib()  — uses python-sgp4 (Brandon Rhodes), the battle-tested reference
#                  implementation with automatic SGP4/SDP4 selection.
#   _sgp4_builtin() — hand-rolled SGP4 (Hoots & Roehrich 1980), near-Earth only.
#
# _sgp4_state() dispatches to _sgp4_lib() when python-sgp4 is installed,
# otherwise falls back to _sgp4_builtin() automatically.


def _sgp4_lib(tle: dict, dt_sec: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    SGP4/SDP4 via python-sgp4 (pip install sgp4).
    Automatically selects SGP4 (near-Earth) or SDP4 (deep-space).
    Requires python-sgp4 to be installed.

    Args:
        tle:    dict with keys 'line1', 'line2', 'epoch' (from _parse_tle_text)
        dt_sec: seconds past TLE epoch

    Returns:
        (r_km, v_km_s) osculating TEME ECI position and velocity
    """
    sat = Satrec.twoline2rv(tle['line1'], tle['line2'])
    epoch = pd.Timestamp(tle['epoch'])
    t     = epoch + pd.Timedelta(seconds=dt_sec)
    jd, fr = sgp4_jday(t.year, t.month, t.day,
                        t.hour, t.minute,
                        float(t.second) + t.microsecond / 1e6)
    e, r, v = sat.sgp4(jd, fr)
    if e != 0:
        raise ValueError(f"python-sgp4 error code {e} at {t}")
    return np.array(r), np.array(v)


def _sgp4_builtin(mean: dict, dt_sec: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Hand-rolled SGP4 fallback (Hoots & Roehrich 1980 / Vallado 2006).
    Near-Earth orbits only (period < 225 min). Used when python-sgp4 is
    not installed.

    Args:
        mean:   dict from _tle_to_mean()
        dt_sec: seconds since TLE epoch

    Returns:
        (r_km, v_km_s) osculating ECI position and velocity
    """
    # SGP4 constants (WGS-72)
    ke   = 0.0743669161   # sqrt(GM) in ER^(3/2)/min
    k2   = 5.413080e-4    # (1/2) J2 RE^2
    k4   = 0.62098875e-6  # -(3/8) J4 RE^4
    A30  = -1.122820e-5   # -J3 RE^3
    CK2  = k2
    CK4  = k4
    ER   = RE             # Earth radius km (used for unit conversion)
    XKMPER = RE           # km per Earth radius

    # Convert mean elements to SGP4 units (Earth radii, radians, minutes)
    xno   = mean['mean_mot'] * 2 * np.pi / 1440.0   # rad/min
    eo    = mean['ecc']
    xincl = np.radians(mean['inc'])
    xnodeo= np.radians(mean['raan'])
    omegao= np.radians(mean['argp'])
    xmo   = np.radians(mean['mean_anom'])
    bstar = mean['bstar']
    tsince= dt_sec / 60.0   # minutes since epoch

    # Recovery of original mean motion (n0'') and semi-major axis (a0'')
    a1    = (ke / xno) ** (2/3)
    thetasq = np.cos(xincl)**2
    x3thm1  = 3 * thetasq - 1.0
    eosq    = eo**2
    betao2  = 1 - eosq
    betao   = np.sqrt(betao2)
    del1    = 1.5 * CK2 * x3thm1 / (a1**2 * betao * betao2)
    ao      = a1 * (1 - del1 * (1/3 + del1 * (1 + 134/81 * del1)))
    delo    = 1.5 * CK2 * x3thm1 / (ao**2 * betao * betao2)
    xnodp   = xno / (1 + delo)          # recovered mean motion (rad/min)
    aodp    = ao / (1 - delo)           # recovered semi-major axis (ER)

    # Initialization
    isimp = 1 if (aodp * (1 - eo)) < (220/XKMPER + 1) else 0

    s     = 78/XKMPER + 1.0
    qoms2t= ((120 - 78) / XKMPER)**4
    perige= (aodp * (1 - eo) - 1) * XKMPER   # perigee height km

    if perige < 156:
        s4 = perige - 78
        if perige <= 98:
            s4 = 20.0
        qoms2t = ((120 - s4) / XKMPER)**4
        s4 = s4 / XKMPER + 1.0
    else:
        s4 = s

    pinvsq  = 1 / (aodp**2 * betao2**2)
    tsi     = 1 / (aodp - s4)
    eta     = aodp * eo * tsi
    etasq   = eta**2
    eeta    = eo * eta
    psisq   = abs(1 - etasq)
    coef    = qoms2t * tsi**4
    coef1   = coef / psisq**3.5
    c2      = (coef1 * xnodp
               * (aodp * (1 + 1.5*etasq + eeta*(4 + etasq))
                  + 0.75*CK2*tsi/psisq*x3thm1*(8 + 3*etasq*(8 + etasq))))
    c1      = bstar * c2
    sinio   = np.sin(xincl)
    a3ovk2  = -A30 / CK2
    c3      = 0.0
    if eo > 1e-4:
        c3 = coef * tsi * a3ovk2 * xnodp * sinio / eo
    x1mth2  = 1 - thetasq
    c4      = (2 * xnodp * coef1 * aodp * betao2
               * (eta*(2 + 0.5*etasq) + eo*(0.5 + 2*etasq)
                  - 2*CK2*tsi/(aodp*psisq)
                  * (-3*x3thm1*(1 - 2*eeta + etasq*(1.5 - 0.5*eeta))
                     + 0.75*x1mth2*(2*etasq - eeta*(1 + etasq))*np.cos(2*omegao))))
    c5 = 2*coef1*aodp*betao2*(1 + 2.75*(etasq + eeta) + eeta*etasq)
    theta2  = thetasq
    x1m5th  = 1 - 5*thetasq
    xmdot   = (xnodp + 0.5*CK2*(-x3thm1)*xnodp*pinvsq/betao
               + CK2**2*xnodp*(13 - 78*thetasq + 137*theta2**2)*pinvsq**2/(16*betao2**2))  # Removed extra 2
    x1mth2  = 1 - thetasq
    omgdot  = (-0.5*CK2*x1m5th*xnodp*pinvsq/betao
               + CK2**2*(7 - 114*thetasq + 395*theta2**2)*xnodp*pinvsq**2/(16*betao2**2)
               + 5*CK4*xnodp*(3 - 36*thetasq + 49*theta2**2)*pinvsq**2/(4*betao2**2))
    xhdot1  = -CK2 * xnodp * pinvsq / betao
    xnodot  = (xhdot1*np.cos(xincl)
               + (CK2**2*(4 - 19*thetasq)*xnodp*pinvsq**2/(2*betao2**2)
                  + 5*CK4*(3 - 7*thetasq)*xnodp*pinvsq**2/(2*betao2**2))*np.cos(xincl))
    omgcof  = bstar * c3 * np.cos(omegao)
    xmcof   = -(2/3) * coef * bstar / eeta if (eo > 1e-4 and abs(eeta) > 1e-10) else 0.0
    xnodcf  = 3.5 * betao2 * xhdot1 * c1
    t2cof   = 1.5 * c1
    xlcof   = 0.125 * a3ovk2 * sinio * (3 + 5*thetasq) / (1 + np.cos(xincl)) if abs(np.cos(xincl)+1) < 1.5e-12 else 0.0
    aycof   = 0.25 * a3ovk2 * sinio
    delmo   = (1 + eta*np.cos(xmo))**3
    sinmo   = np.sin(xmo)
    x7thm1  = 7*thetasq - 1

    # Update for secular gravity and atmospheric drag (simplified)
    xmdf    = xmo + xmdot * tsince
    omgadf  = omegao + omgdot * tsince
    xnoddf  = xnodeo + xnodot * tsince
    omega   = omgadf
    xmp     = xmdf
    tsq     = tsince**2
    xnode   = xnoddf + xnodcf * tsq
    tempa   = 1 - c1 * tsince
    tempe   = bstar * c4 * tsince
    templ   = t2cof * tsq

    if not isimp:
        delomg = omgcof * tsince
        delm   = xmcof * ((1 + eta*np.cos(xmdf))**3 - delmo)
        temp   = delomg + delm
        xmp    = xmdf + temp
        omega  = omgadf - temp
        tcube  = tsq * tsince
        tfour  = tsince * tcube
        tempa  = tempa - (c3*np.cos(omegao) + c5*(np.sin(xmp) - sinmo))*tsince  # simplified
        tempe  = tempe + bstar * c5 * (np.sin(xmp) - sinmo)
        templ  = templ + t2cof*tsq   # simplified; full version adds higher order

    a    = aodp * tempa**2
    e    = max(eo - tempe, 1e-6)
    xl   = xmp + omega + xnode + xnodp * templ
    beta = np.sqrt(1 - e**2)
    xn   = ke / a**1.5

    # Long period periodics
    axn  = e * np.cos(omega)
    temp = 1 / (a * beta**2)
    xll  = temp * xlcof * axn
    aynl = temp * aycof
    xlt  = xl + xll
    ayn  = e * np.sin(omega) + aynl

    # Solve Kepler's equation (iterative)
    capu  = (xlt - xnode) % (2 * np.pi)
    epw   = capu
    for _ in range(10):
        sinepw = np.sin(epw)
        cosepw = np.cos(epw)
        f      = capu - (epw - axn*sinepw + ayn*cosepw)
        fdot   = 1 - axn*cosepw - ayn*sinepw
        epw   += f / fdot

    sinepw = np.sin(epw)
    cosepw = np.cos(epw)

    # Short period preliminary quantities
    ecose = axn*cosepw + ayn*sinepw
    esine = axn*sinepw - ayn*cosepw
    elsq  = axn**2 + ayn**2
    temp  = 1 - elsq
    pl    = a * temp
    r_er  = a * (1 - ecose)
    rdot  = ke * np.sqrt(a) * esine / r_er
    rfdot = ke * np.sqrt(pl) / r_er
    temp2 = a / r_er
    betal = np.sqrt(temp)
    if abs(1 + betal) > 1e-10:
        cosu  = temp2 * (cosepw - axn + ayn*esine / (1 + betal))
        sinu  = temp2 * (sinepw - ayn - axn*esine / (1 + betal))
    else:
        cosu  = temp2 * (cosepw - axn)
        sinu  = temp2 * (sinepw - ayn)
    u     = np.arctan2(sinu, cosu)
    sin2u = 2 * sinu * cosu
    cos2u = 1 - 2 * sinu**2

    # Update with short period periodics
    temp  = CK2 / pl**2
    temp1 = CK2 / pl
    rk    = r_er * (1 - 1.5*temp*betal*x3thm1) + 0.5*temp1*x1mth2*cos2u
    uk    = u - 0.25*temp*x7thm1*sin2u
    xnodek = xnode + 1.5*temp*np.cos(xincl)*sin2u
    xinck  = xincl + 1.5*temp*np.cos(xincl)*np.sin(xincl)*cos2u
    rdotk  = rdot - xn*temp1*x1mth2*sin2u
    rfdotk = rfdot + xn*temp1*(x1mth2*cos2u + 1.5*x3thm1)

    # Orientation vectors
    sinuk  = np.sin(uk)
    cosuk  = np.cos(uk)
    sinik  = np.sin(xinck)
    cosik  = np.cos(xinck)
    sinnok = np.sin(xnodek)
    cosnok = np.cos(xnodek)
    xmx    = -sinnok * cosik
    xmy    =  cosnok * cosik
    ux     = xmx*sinuk + cosnok*cosuk
    uy     = xmy*sinuk + sinnok*cosuk
    uz     = sinik * sinuk
    vx     = xmx*cosuk - cosnok*sinuk
    vy     = xmy*cosuk - sinnok*sinuk
    vz     = sinik * cosuk

    # Position (km) and velocity (km/s)
    r_km = rk * XKMPER * np.array([ux, uy, uz])
    v_kms= (rdotk * ux + rfdotk * vx,
            rdotk * uy + rfdotk * vy,
            rdotk * uz + rfdotk * vz)
    v_km = np.array(v_kms) * XKMPER / 60.0   # ER/min → km/s

    return r_km, v_km


def _sgp4_state(tle: dict, dt_sec: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Dispatcher: use python-sgp4 if available, else fall back to hand-rolled SGP4.

    Args:
        tle:    dict with keys 'line1', 'line2', 'epoch', and optionally
                pre-parsed 'mean' (from _tle_to_mean). Both are accepted.
        dt_sec: seconds past TLE epoch

    Returns:
        (r_km, v_km_s) osculating ECI position and velocity
    """
    if _SGP4_LIB:
        return _sgp4_lib(tle, dt_sec)
    else:
        mean = tle.get('mean') or _tle_to_mean(tle['line1'], tle['line2'])
        return _sgp4_builtin(mean, dt_sec)


# ── 4. OD Propagator (RK4 with J2 + Drag) ────────────────────────────────────

def _atmo_density(alt_km: float) -> float:
    """
    US Standard Atmosphere 1976 piecewise exponential density (kg/m^3).
    Much more accurate than a single scale height, especially above 100 km.
    """
    if alt_km >= _ATMO_LAYERS[-1, 0]:
        base_alt, base_rho, H = _ATMO_LAYERS[-1]
    else:
        idx = np.searchsorted(_ATMO_LAYERS[:, 0], alt_km, side='right') - 1
        idx = max(0, idx)
        base_alt, base_rho, H = _ATMO_LAYERS[idx]
    return base_rho * np.exp(-(alt_km - base_alt) / H)   # kg/m^3


def _bstar_to_drag(bstar: float) -> float:
    """
    Convert TLE B* (1/Earth-radius) to effective CD*Am (m^2/kg).

    SGP4 defines B* in units of 1/Earth-radius (not 1/km):
        B* = (CD * A) / (2 * m) * rho_0
    where rho_0 = 2.461e-5 kg/m^2 (SGP4 reference density at 0 km in ER units).

    To use B* in our drag model we solve for CD*Am:
        CD * Am = 2 * B* [1/ER] / (rho_0 [kg/m^2/ER])
               = 2 * B* / 2.461e-5   (m^2/kg)

    Note: B* from _parse_bstar is already in 1/ER (TLE native units).
    """
    RHO0_SGP4 = 2.461e-5   # kg/m^2 per Earth radius (SGP4 reference density)
    return 2.0 * abs(bstar) / RHO0_SGP4   # m^2/kg


def _derivatives(state, CD_Am_eff: float):
    """
    Equations of motion: two-body + J2 oblateness + atmospheric drag.

    Drag improvements over naive implementation:
      - US Standard Atmosphere 1976 piecewise density (altitude-dependent scale height)
      - Velocity relative to the co-rotating atmosphere (Earth rotation correction)

    Args:
        state:      [x, y, z, vx, vy, vz] in km and km/s (ECI frame)
        CD_Am_eff:  effective CD * A/m in m^2/kg (from B* or CD*Am)
    """
    x, y, z, vx, vy, vz = state
    r = np.sqrt(x**2 + y**2 + z**2)

    # Two-body gravity
    a_grav = -MU / r**3 * np.array([x, y, z])

    # J2 oblateness perturbation
    fac  = 1.5 * J2 * MU * RE**2 / r**5
    z_r2 = (z / r)**2
    a_J2 = fac * np.array([
        x * (5*z_r2 - 1),
        y * (5*z_r2 - 1),
        z * (5*z_r2 - 3)
    ])

    # Atmospheric drag — velocity relative to rotating atmosphere
    # The atmosphere co-rotates with Earth: v_atmo = omega_E × r (ECI)
    # Only x,y components are affected (rotation around z-axis)
    v_rel = np.array([
        vx + OMEGA_E * y,   # vx_inertial - (-omega*y) = vx + omega*y
        vy - OMEGA_E * x,   # vy_inertial - ( omega*x) = vy - omega*x
        vz
    ])
    v_rel_mag = np.linalg.norm(v_rel)

    alt_km = r - RE
    rho_si = _atmo_density(alt_km)          # kg/m^3
    rho_km = rho_si * 1e9                   # convert to kg/km^3
    # CD_Am_eff is m^2/kg; multiply by 1e-6 to convert m^2 → km^2
    a_drag = -0.5 * CD_Am_eff * 1e-6 * rho_km * v_rel_mag * v_rel   # km/s^2

    acc = a_grav + a_J2 + a_drag
    return np.array([vx, vy, vz, acc[0], acc[1], acc[2]])


def propagate_od(tle: dict, dt_sec: float,
                 CD: float = 2.2, Am: float = 0.01,
                 use_bstar: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Propagate a TLE forward by dt_sec seconds using SGP4 + RK4 OD integration.

    Pipeline:
      1. SGP4 at t=0 → osculating ECI state at TLE epoch
         (uses python-sgp4 if installed, else hand-rolled fallback)
      2. RK4 integration with J2 + atmospheric drag forward by dt_sec

    Using SGP4 for step 1 is essential — TLE mean elements are NOT osculating
    Keplerian elements. The SGP4 conversion removes short-period and secular
    terms to produce a physically correct initial state.

    Drag calibration:
      - use_bstar=True: B* converted to effective CD*Am (per-epoch calibrated)
      - use_bstar=False: generic CD * Am used for all epochs

    Args:
        tle:       dict with 'line1', 'line2', 'epoch' (from _parse_tle_text)
        dt_sec:    propagation duration in seconds
        CD:        fallback drag coefficient
        Am:        fallback area-to-mass ratio m^2/kg
        use_bstar: if True, prefer B*-derived drag

    Returns:
        (position_km, velocity_km_s) at epoch + dt_sec
    """
    # Step 1: SGP4 → osculating ECI state at epoch
    r0, v0 = _sgp4_state(tle, dt_sec=0.0)

    # Resolve effective drag coefficient from B* or fallback CD/Am
    mean  = tle.get('mean') or _tle_to_mean(tle['line1'], tle['line2'])
    bstar = mean.get('bstar', 0.0)
    CD_Am_eff = _bstar_to_drag(bstar) if (use_bstar and bstar > 1e-10) else CD * Am

    # Step 2: RK4 integration from osculating state
    state   = np.concatenate([r0, v0])
    n_steps = max(10, int(dt_sec / 60))
    h       = dt_sec / n_steps

    for _ in range(n_steps):
        k1 = _derivatives(state,           CD_Am_eff)
        k2 = _derivatives(state + h/2*k1,  CD_Am_eff)
        k3 = _derivatives(state + h/2*k2,  CD_Am_eff)
        k4 = _derivatives(state + h*k3,    CD_Am_eff)
        state += (h / 6) * (k1 + 2*k2 + 2*k3 + k4)

    return state[:3], state[3:]


# ── 5. Space Weather (Kp Index) ───────────────────────────────────────────────

def _parse_kp_text(text: str, file_type: str = 'nowcast') -> pd.DataFrame:
    """
    Parse GFZ Potsdam Kp text file into a DataFrame.

    file_type 'nowcast': one row per 3hr interval, Kp at column index 7
    file_type 'daily':   one row per day, Kp values at column indices 7-14
    """
    records = []
    for line in text.splitlines():
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split()
        try:
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            if file_type == 'nowcast':
                hour = float(parts[3])
                kp   = float(parts[7])
                if kp < 0:
                    continue
                records.append({
                    'datetime': pd.Timestamp(year, month, day,
                                             int(hour), int((hour % 1) * 60)),
                    'kp': kp
                })
            elif file_type == 'daily':
                for hour_idx in range(8):
                    kp = float(parts[7 + hour_idx])
                    if kp < 0:
                        continue
                    records.append({
                        'datetime': pd.Timestamp(year, month, day, hour_idx * 3, 0),
                        'kp': kp
                    })
        except (ValueError, IndexError):
            continue
    return pd.DataFrame(records)


def _fetch_kp_from_source(since_year: int = None) -> tuple[pd.DataFrame, str]:
    """Download Kp index from GFZ Potsdam."""
    if since_year is not None and since_year >= 2024:
        url       = "https://kp.gfz-potsdam.de/app/files/Kp_ap_nowcast.txt"
        file_type = 'nowcast'
        print("  Downloading Kp nowcast file...")
    else:
        url       = "https://kp.gfz-potsdam.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt"
        file_type = 'daily'
        print("  Downloading full Kp historical file...")
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    return _parse_kp_text(response.text, file_type=file_type), response.text


def get_kp_index(start_date: str, end_date: str, data_dir: str = "./data") -> pd.DataFrame:
    """
    Returns Kp index for a date range, using local cache if fresh.
    Saves kp_index.csv and kp_index.txt to data_dir.
    Auto-detects and re-downloads corrupt files (values outside 0-9).
    """
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "kp_index.csv")
    txt_path = os.path.join(data_dir, "kp_index.txt")

    if os.path.exists(csv_path):
        kp_df    = pd.read_csv(csv_path, parse_dates=['datetime'])
        latest   = kp_df['datetime'].max()
        age_days = (datetime.now() - latest.to_pydatetime().replace(tzinfo=None)).days

        # Sanity check
        if kp_df['kp'].max() > 9 or kp_df['kp'].min() < 0:
            print(f"WARNING: Corrupt Kp file detected. Re-downloading...")
            os.remove(csv_path)
            if os.path.exists(txt_path):
                os.remove(txt_path)
            kp_df, raw_txt = _fetch_kp_from_source(since_year=None)
            with open(txt_path, 'w') as f: f.write(raw_txt)
            kp_df.to_csv(csv_path, index=False)
            print(f"Saved {txt_path} and {csv_path}")

        elif age_days <= 1:
            print(f"Loaded local Kp file (latest: {latest})")

        else:
            print(f"Kp file is {age_days} days old. Updating...")
            new_df, new_txt = _fetch_kp_from_source(since_year=latest.year)
            with open(txt_path, 'a') as f: f.write(new_txt)
            kp_df = pd.concat([
                kp_df[kp_df['datetime'] < new_df['datetime'].min()], new_df
            ]).drop_duplicates('datetime').sort_values('datetime').reset_index(drop=True)
            kp_df.to_csv(csv_path, index=False)
            print(f"Updated {csv_path} (up to {kp_df['datetime'].max()})")
    else:
        print("No local Kp file. Downloading full historical file...")
        kp_df, raw_txt = _fetch_kp_from_source(since_year=None)
        with open(txt_path, 'w') as f: f.write(raw_txt)
        kp_df.to_csv(csv_path, index=False)
        print(f"Saved {txt_path} and {csv_path} ({len(kp_df):,} records)")

    return kp_df[(kp_df['datetime'] >= start_date) &
                 (kp_df['datetime'] <= end_date)].reset_index(drop=True)


def _get_kp_for_epoch(epoch: pd.Timestamp, kp_df: pd.DataFrame) -> float:
    """Returns Kp for the exact 3-hour interval containing the epoch."""
    mask  = (kp_df['datetime'] <= epoch) & \
            (kp_df['datetime'] + pd.Timedelta(hours=3) > epoch)
    match = kp_df[mask]
    if not match.empty:
        return match.iloc[0]['kp']
    idx = (kp_df['datetime'] - epoch).abs().argmin()
    return kp_df.iloc[idx]['kp']


# ── 6. Main: OD-Based Manoeuvre Detection ────────────────────────────────────

def analyse_maneuvers_od(catnr: int, username: str, password: str,
                         data_dir: str = "./data",
                         output_dir: str = "./output",
                         sigma_multiplier: float = 5.0,
                         kp_threshold: float = 5.0,
                         max_gap_days: float = 10.0,
                         CD: float = 2.2,
                         Am: float = 0.01,
                         use_bstar: bool = True,
                         bstar_noise_max: float = 1e-3,
                         delta_bstar_threshold: float = 5e-5) -> pd.DataFrame:
    """
    OD-based manoeuvre detection from Space-Track TLE history.

    For each consecutive TLE pair:
      1. Filter out noisy TLEs where |B*| > bstar_noise_max
      2. Run SGP4 at epoch → osculating ECI state
      3. Propagate forward with RK4 (J2 + drag calibrated from B*)
      4. Compare predicted vs actual state at next epoch
      5. Compute gap-normalised velocity residual (m/s per day)
      6. Derive adaptive threshold from the data: median + sigma_multiplier × MAD
      7. Flag B* step-changes as secondary signal
      8. Cross-check with Kp to classify as confirmed or uncertain

    Threshold rationale:
      The velocity residual grows roughly linearly with gap length due to
      integrator and atmosphere model errors. Normalising by gap_days gives a
      stationary noise floor. The adaptive threshold is set at:
          median(vel_res_per_day) + sigma_multiplier × MAD
      where MAD = median absolute deviation, a robust spread estimator.
      Epochs above this are statistically anomalous — i.e. manoeuvres.

    B* is used in three ways:
      - Propagator drag: B* replaces generic CD/Am for per-epoch calibrated drag
      - Noise filter: epochs with |B*| > bstar_noise_max are skipped (bad TLE fits)
      - Step-change signal: sudden jumps in B* indicate an unmodelled force

    Args:
        catnr:                   NORAD catalog number
        username:                Space-Track username (email)
        password:                Space-Track password
        data_dir:                Folder for local TLE and Kp cache
        output_dir:              Folder for plot output files
        sigma_multiplier:        Threshold = median + N × MAD (default 5)
        kp_threshold:            Kp above which space weather is disturbed
        max_gap_days:            Skip epoch pairs with gaps larger than this
        CD:                      Fallback drag coefficient (when B* unavailable)
        Am:                      Fallback area-to-mass ratio m^2/kg
        use_bstar:               Use B* for drag in the propagator (recommended)
        bstar_noise_max:         Skip TLEs where |B*| > this value (noisy fit)
        delta_bstar_threshold:   |ΔB*| above this flags a manoeuvre signal

    Returns:
        DataFrame with one row per TLE pair and manoeuvre classification
    """
    # ── Load TLEs ─────────────────────────────────────────────────────────────
    tles = fetch_tle_history(catnr, username, password, data_dir=data_dir)
    if len(tles) < 2:
        raise ValueError(f"Not enough TLEs found for CATNR {catnr}")

    # ── B* noise filter: skip TLEs with anomalously large |B*| ───────────────
    n_before = len(tles)
    tles_filtered = []
    n_skipped_bstar = 0
    for tle in tles:
        el = _tle_to_mean(tle['line1'], tle['line2'])
        tle['mean'] = el   # cache for later use
        if abs(el['bstar']) > bstar_noise_max:
            n_skipped_bstar += 1
        else:
            tles_filtered.append(tle)
    if n_skipped_bstar:
        print(f"B* noise filter: skipped {n_skipped_bstar} TLEs with |B*| > {bstar_noise_max} "
              f"({n_before - n_skipped_bstar} remaining)")
    tles = tles_filtered

    if len(tles) < 2:
        raise ValueError(f"Not enough valid TLEs after B* noise filter for CATNR {catnr}")

    # ── Propagate and compute residuals ───────────────────────────────────────
    print(f"Propagating {len(tles)-1} TLE pairs...")
    results = []

    for i in range(len(tles) - 1):
        tle_now  = tles[i]
        tle_next = tles[i + 1]

        epoch_now  = pd.Timestamp(tle_now['epoch'])
        epoch_next = pd.Timestamp(tle_next['epoch'])
        gap_sec    = (epoch_next - epoch_now).total_seconds()
        gap_days   = gap_sec / 86400

        if gap_days > max_gap_days or gap_days <= 0:
            continue

        # Cache parsed mean elements in tle dict for reuse
        tle_now['mean']  = _tle_to_mean(tle_now['line1'],  tle_now['line2'])
        tle_next['mean'] = _tle_to_mean(tle_next['line1'], tle_next['line2'])
        mean_now  = tle_now['mean']
        mean_next = tle_next['mean']

        # Propagate from epoch_now → epoch_next using SGP4 + RK4 OD
        try:
            r_pred, v_pred = propagate_od(tle_now, gap_sec, CD=CD, Am=Am,
                                          use_bstar=use_bstar)
        except Exception as e:
            print(f"  Propagation failed at {epoch_now}: {e}")
            continue

        # Actual state at epoch_next via SGP4 at its own epoch (dt=0)
        try:
            r_actual, v_actual = _sgp4_state(tle_next, dt_sec=0.0)
        except Exception as e:
            print(f"  SGP4 failed at {epoch_next}: {e}")
            continue

        pos_residual_km = np.linalg.norm(r_pred - r_actual)
        vel_residual_ms = np.linalg.norm(v_pred - v_actual) * 1000   # m/s

        # Osculating SMA from vis-viva: a = 1 / (2/r - v^2/mu)
        r_now, v_now = _sgp4_state(tle_now, dt_sec=0.0)
        sma_now  = 1.0 / (2.0/np.linalg.norm(r_now)  - np.linalg.norm(v_now)**2  / MU)
        sma_next = 1.0 / (2.0/np.linalg.norm(r_actual) - np.linalg.norm(v_actual)**2 / MU)

        delta_sma   = sma_next - sma_now       # signed, for display
        delta_ecc   = mean_next['ecc']   - mean_now['ecc']
        delta_inc   = mean_next['inc']   - mean_now['inc']
        delta_raan  = mean_next['raan']  - mean_now['raan']
        delta_raan  = ((delta_raan + 180) % 360) - 180
        delta_bstar = mean_next['bstar'] - mean_now['bstar']

        results.append({
            'epoch_from':        epoch_now,
            'epoch_to':          epoch_next,
            'gap_days':          round(gap_days, 4),
            'sma':               round(sma_now, 3),
            'delta_sma':         round(delta_sma, 3),         # signed, for display
            'abs_delta_sma':     round(abs(delta_sma), 3),    # absolute, for threshold comparisons
            'delta_ecc':         round(delta_ecc, 6),
            'delta_inc':         round(delta_inc, 4),
            'delta_raan':        round(delta_raan, 4),
            'bstar':             round(mean_now['bstar'], 8),
            'delta_bstar':       round(delta_bstar, 8),
            'pos_residual_km':   round(pos_residual_km, 3),
            'vel_residual_ms':   round(vel_residual_ms, 4),
        })

    result_df = pd.DataFrame(results)
    print(f"Computed {len(result_df)} epoch pairs")

    # ── Kp lookup ─────────────────────────────────────────────────────────────
    start = result_df['epoch_from'].min().strftime('%Y-%m-%d')
    end   = result_df['epoch_to'].max().strftime('%Y-%m-%d')
    kp_df = get_kp_index(start, end, data_dir=data_dir)

    result_df['kp'] = result_df['epoch_from'].apply(
        lambda ep: _get_kp_for_epoch(ep, kp_df)
    )

    # ── Adaptive threshold from data distribution ────────────────────────────
    # Normalise by gap length — residual grows linearly with gap_days
    result_df['vel_res_per_day'] = result_df['vel_residual_ms'] / result_df['gap_days']

    # Exclude inf/nan values from threshold computation — inf residuals arise from
    # degenerate propagations and must not corrupt the robust statistics.
    finite_mask = np.isfinite(result_df['vel_res_per_day'])
    n_inf = (~finite_mask).sum()
    if n_inf > 0:
        print(f"WARNING: {n_inf} epoch pair(s) have non-finite vel_res_per_day "
              f"and will be excluded from threshold computation.")

    vel_res_finite = result_df.loc[finite_mask, 'vel_res_per_day']

    # Robust statistics: median and MAD (median absolute deviation)
    # MAD is resistant to outliers (i.e. actual manoeuvres won't inflate it)
    median_vr  = vel_res_finite.median()
    mad_vr     = (vel_res_finite - median_vr).abs().median()
    threshold_per_day = median_vr + sigma_multiplier * mad_vr

    # Absolute threshold: reconstruct from per-day value × typical gap
    # Use median gap so the absolute threshold shown in plots is meaningful
    median_gap            = result_df['gap_days'].median()
    vel_residual_threshold = threshold_per_day * median_gap

    # ── Manoeuvre classification ─────────────────────────────────────────────
    result_df['bad_space_weather'] = result_df['kp'] >= kp_threshold

    # OD signal: gap-normalised residual above adaptive threshold.
    # Non-finite residuals are not compared against the threshold — they are
    # flagged separately so they do not masquerade as manoeuvre detections.
    od_signal = finite_mask & (result_df['vel_res_per_day'] > threshold_per_day)
    result_df['vel_res_inf'] = ~finite_mask   # True where residual is non-finite

    # B* step-change — secondary corroborating signal
    bstar_signal = result_df['delta_bstar'].abs() > delta_bstar_threshold

    result_df['od_flag']            = od_signal
    result_df['bstar_flag']         = bstar_signal
    result_df['likely_maneuver']    = od_signal | bstar_signal
    result_df['confirmed_maneuver'] = result_df['likely_maneuver'] & ~result_df['bad_space_weather']
    result_df['uncertain_maneuver'] = result_df['likely_maneuver'] &  result_df['bad_space_weather']

    # ── Summary ───────────────────────────────────────────────────────────────
    alt_km       = result_df['sma'].mean() - RE
    bstar_median = result_df['bstar'].median()
    bstar_used   = "B* (TLE-calibrated)" if use_bstar else f"CD={CD} × Am={Am}"
    print(f"\n── OD Manoeuvre Analysis: CATNR {catnr} ────────────────────────")
    print(f"  TLE pairs analysed:        {len(result_df)}")
    print(f"  Mean altitude:             {alt_km:.0f} km")
    print(f"  Drag source:               {bstar_used}")
    print(f"  Median B*:                 {bstar_median:.2e}")
    print(f"  Sigma multiplier:          {sigma_multiplier}×")
    print(f"  Noise floor (median):      {median_vr:.4f} m/s/day")
    print(f"  MAD spread:                {mad_vr:.4f} m/s/day")
    print(f"  Adaptive threshold:        {threshold_per_day:.4f} m/s/day  "
          f"(≈ {vel_residual_threshold:.4f} m/s at median gap {median_gap:.3f} days)")
    print(f"  ΔB* threshold:             {delta_bstar_threshold:.1e}")
    print(f"  Kp threshold:              {kp_threshold}")
    print(f"  OD-flagged:                {od_signal.sum()}")
    print(f"  Non-finite vel residuals:  {result_df['vel_res_inf'].sum()}")
    print(f"  B*-flagged:                {bstar_signal.sum()}")
    print(f"  Likely manoeuvres:         {result_df['likely_maneuver'].sum()}")
    print(f"  Confirmed manoeuvres:      {result_df['confirmed_maneuver'].sum()}")
    print(f"  Uncertain (bad weather):   {result_df['uncertain_maneuver'].sum()}")
    print(f"  Median vel residual:       {result_df['vel_residual_ms'].median():.4f} m/s")
    print(f"  Max  vel residual:         {result_df['vel_residual_ms'].max():.4f} m/s")

    def _print_rows(rows, label):
        for _, r in rows.iterrows():
            signals = []
            if r['od_flag']:    signals.append('OD')
            if r['bstar_flag']: signals.append('B*')
            print(f"  {r['epoch_from']}  →  {r['epoch_to']}  "
                  f"(gap: {r['gap_days']:.3f} days)  "
                  f"vel_res: {r['vel_residual_ms']:.4f} m/s  "
                  f"ΔB*: {r['delta_bstar']:+.2e}  "
                  f"ΔSMA: {r['delta_sma']:+.3f} km  "
                  f"ΔEcc: {r['delta_ecc']:+.2e}  "
                  f"ΔInc: {r['delta_inc']:+.4f}°  "
                  f"Kp: {r['kp']:.1f}  "
                  f"signal: {'+'.join(signals)}  [{label}]")

    if result_df['confirmed_maneuver'].any():
        print(f"\n── Confirmed Manoeuvre Epochs ───────────────────────────────────")
        _print_rows(result_df[result_df['confirmed_maneuver']], 'CONFIRMED')

    if result_df['uncertain_maneuver'].any():
        print(f"\n── Uncertain Epochs (possible manoeuvre, bad space weather) ─────")
        _print_rows(result_df[result_df['uncertain_maneuver']], 'UNCERTAIN')

    # ── Plot ──────────────────────────────────────────────────────────────────
    _plot_results(result_df, catnr,
                  threshold_per_day=threshold_per_day,
                  median_gap=median_gap,
                  kp_threshold=kp_threshold,
                  delta_bstar_threshold=delta_bstar_threshold,
                  output_dir=output_dir)

    return result_df


def _plot_results(result_df: pd.DataFrame, catnr: int,
                  threshold_per_day: float, median_gap: float,
                  kp_threshold: float,
                  delta_bstar_threshold: float = 5e-5,
                  output_dir: str = "./output") -> None:
    """Plot OD residuals, B* history, element deltas, and Kp with manoeuvre markers."""
    os.makedirs(output_dir, exist_ok=True)

    confirmed = result_df[result_df['confirmed_maneuver']]
    uncertain = result_df[result_df['uncertain_maneuver']]

    fig, axes = plt.subplots(7, 1, figsize=(13, 20), sharex=True)
    fig.suptitle(f'OD Manoeuvre Detection — CATNR {catnr}', fontsize=13, y=0.98)

    def _scatter_flags(ax, col):
        if not confirmed.empty:
            ax.scatter(confirmed['epoch_to'], confirmed[col],
                       color='red', zorder=5, s=30, label='Confirmed')
        if not uncertain.empty:
            ax.scatter(uncertain['epoch_to'], uncertain[col],
                       color='orange', zorder=5, marker='^', s=30, label='Uncertain')

    # Panel 1: Gap-normalised velocity residual with adaptive threshold
    axes[0].plot(result_df['epoch_to'], result_df['vel_res_per_day'],
                 marker='o', markersize=2, color='darkblue', linewidth=0.8,
                 label='Vel residual / gap_days')
    axes[0].axhline(threshold_per_day, color='red', linestyle='--', linewidth=1,
                    label=f'Adaptive threshold ({threshold_per_day:.3f} m/s/day)')
    _scatter_flags(axes[0], 'vel_res_per_day')
    axes[0].set_ylabel('Vel Residual(m/s per day)')
    axes[0].set_yscale('log')
    axes[0].legend(fontsize=8, loc='upper right')

    # Panel 2: Raw velocity residual (for reference)
    axes[1].plot(result_df['epoch_to'], result_df['vel_residual_ms'],
                 marker='o', markersize=2, color='steelblue', linewidth=0.8)
    _scatter_flags(axes[1], 'vel_residual_ms')
    axes[1].set_ylabel('Vel Residual(m/s raw)')
    axes[1].set_yscale('log')
    axes[1].legend(fontsize=8, loc='upper right')

    # Panel 3: B* value over time
    axes[2].plot(result_df['epoch_to'], result_df['bstar'],
                 marker='o', markersize=2, color='saddlebrown', linewidth=0.8)
    _scatter_flags(axes[2], 'bstar')
    axes[2].set_ylabel('B* (1/ER)')
    axes[2].legend(fontsize=8, loc='upper right')

    # Panel 4: ΔB* step-change signal
    axes[3].plot(result_df['epoch_to'], result_df['delta_bstar'],
                 marker='o', markersize=2, color='chocolate', linewidth=0.8)
    axes[3].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[3].axhline( delta_bstar_threshold, color='red', linestyle=':', linewidth=1,
                     label=f'±{delta_bstar_threshold:.1e}')
    axes[3].axhline(-delta_bstar_threshold, color='red', linestyle=':', linewidth=1)
    _scatter_flags(axes[3], 'delta_bstar')
    axes[3].set_ylabel('ΔB* (1/ER)')
    axes[3].legend(fontsize=8, loc='upper right')

    # Panel 5: ΔSMA
    axes[4].plot(result_df['epoch_to'], result_df['delta_sma'],
                 marker='o', markersize=2, color='darkorange', linewidth=0.8)
    axes[4].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    _scatter_flags(axes[4], 'delta_sma')
    axes[4].set_ylabel('ΔSMA (km)')
    axes[4].legend(fontsize=8, loc='upper right')

    # Panel 6: ΔEccentricity
    axes[5].plot(result_df['epoch_to'], result_df['delta_ecc'],
                 marker='o', markersize=2, color='green', linewidth=0.8)
    axes[5].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    _scatter_flags(axes[5], 'delta_ecc')
    axes[5].set_ylabel('ΔEccentricity')
    axes[5].legend(fontsize=8, loc='upper right')

    # Panel 7: Kp index
    axes[6].plot(result_df['epoch_to'], result_df['kp'],
                 marker='o', markersize=2, color='purple', linewidth=0.8)
    axes[6].axhline(kp_threshold, color='red', linestyle='--', linewidth=1,
                    label=f'Kp threshold ({kp_threshold})')
    axes[6].fill_between(result_df['epoch_to'], result_df['kp'], kp_threshold,
                         where=result_df['kp'] >= kp_threshold,
                         color='red', alpha=0.2, label='Disturbed')
    axes[6].set_ylabel('Kp')
    axes[6].set_xlabel('Epoch')
    axes[6].legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'od_maneuver_detection_{catnr}.png')
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"Plot saved to {plot_path}")


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import getpass
    import os as _os

    CATNR    = int(_os.environ.get("CATNR", "60547"))
    USERNAME = _os.environ.get("SPACETRACK_USER", "")
    PASSWORD = _os.environ.get("SPACETRACK_PASS", "")

    if not USERNAME:
        USERNAME = input("Space-Track username (email): ").strip()
    if not PASSWORD:
        PASSWORD = getpass.getpass("Space-Track password: ")

    df_results = analyse_maneuvers_od(
        catnr                  = CATNR,
        username               = USERNAME,
        password               = PASSWORD,
        data_dir               = "./data",
        output_dir             = "./output",
        sigma_multiplier       = 5.0,       # threshold = median + N×MAD of vel_res/day
        kp_threshold           = 5.0,
        max_gap_days           = 10.0,
        CD                     = 2.2,       # fallback drag coeff (when B* unavailable)
        Am                     = 0.01,      # fallback area-to-mass m^2/kg
        use_bstar              = True,      # use TLE B* for drag calibration
        bstar_noise_max        = 1e-3,      # skip TLEs with |B*| above this
        delta_bstar_threshold  = 5e-5,      # ΔB* step to flag as manoeuvre signal
    )

    print(df_results[['epoch_from', 'epoch_to', 'gap_days',
                       'vel_residual_ms', 'pos_residual_km',
                       'bstar', 'delta_bstar',
                       'delta_sma', 'abs_delta_sma', 'delta_ecc',
                       'od_flag', 'bstar_flag', 'vel_res_inf',
                       'kp', 'confirmed_maneuver', 'uncertain_maneuver']]
          .to_string(index=False))
