"""
main.py
-------
Entry point for satellite manoeuvre detection.

Runs one or both analysis methods for a given NORAD catalog number:
  - Kepler : CelesTrak orbital element deltas (no account required)
  - OD     : Space-Track TLE history + SGP4/RK4 propagation (account required)

Usage:
    # Run both methods (default)
    python main.py

    # Run a specific method
    python main.py --method kepler
    python main.py --method od

    # Override satellite and folders
    python main.py --catnr 25544 --data-dir ./data --output-dir ./output

    # Credentials via env vars (recommended over interactive prompt)
    CATNR=60547 SPACETRACK_USER=you@example.com SPACETRACK_PASS=secret python main.py
"""

import argparse
import getpass
import os
import sys


# ── Argument Parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Satellite manoeuvre detection — Kepler and/or OD methods.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--method",
        choices=["kepler", "od", "both"],
        default="both",
        help="Analysis method to run.",
    )
    parser.add_argument(
        "--catnr",
        type=int,
        default=int(os.environ.get("CATNR", "60547")),
        help="NORAD catalog number. Also reads CATNR env var.",
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Folder for cached TLE and Kp data files.",
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Folder for output plots.",
    )

    # OD-specific options
    od = parser.add_argument_group("OD method options")
    od.add_argument(
        "--sigma",
        type=float,
        default=5.0,
        dest="sigma_multiplier",
        help="Adaptive threshold = median + N×MAD of gap-normalised vel residual.",
    )
    od.add_argument(
        "--max-gap",
        type=float,
        default=10.0,
        dest="max_gap_days",
        help="Skip TLE pairs with epoch gap larger than this (days).",
    )
    od.add_argument(
        "--no-bstar",
        action="store_false",
        dest="use_bstar",
        help="Disable B*-calibrated drag; use generic CD/Am instead.",
    )

    # Kepler-specific options
    kep = parser.add_argument_group("Kepler method options")
    kep.add_argument(
        "--delta-sma",
        type=float,
        default=0.1,
        dest="delta_sma_threshold",
        help="SMA change threshold to flag a manoeuvre (km).",
    )
    kep.add_argument(
        "--delta-ecc",
        type=float,
        default=5e-5,
        dest="delta_ecc_threshold",
        help="Eccentricity change threshold to flag a manoeuvre.",
    )

    # Shared options
    parser.add_argument(
        "--kp-threshold",
        type=float,
        default=5.0,
        dest="kp_threshold",
        help="Kp index above which space weather is considered disturbed.",
    )

    return parser.parse_args()


# ── Credential Resolution ─────────────────────────────────────────────────────

def resolve_credentials() -> tuple[str, str]:
    """
    Resolve Space-Track credentials from env vars, falling back to
    interactive prompts. Returns (username, password).
    """
    username = os.environ.get("SPACETRACK_USER", "")
    password = os.environ.get("SPACETRACK_PASS", "")

    if not username:
        username = input("Space-Track username (email): ").strip()
    if not password:
        password = getpass.getpass("Space-Track password: ")

    return username, password


# ── Runners ───────────────────────────────────────────────────────────────────

def run_kepler(args: argparse.Namespace):
    from orbit_analysis_kepler import analyse_maneuvers_kepler

    print("\n" + "═" * 60)
    print(f"  Kepler Analysis — CATNR {args.catnr}")
    print("═" * 60)

    df = analyse_maneuvers_kepler(
        catnr               = args.catnr,
        data_dir            = args.data_dir,
        output_dir          = args.output_dir,
        delta_sma_threshold = args.delta_sma_threshold,
        delta_ecc_threshold = args.delta_ecc_threshold,
        kp_threshold        = args.kp_threshold,
    )

    print("\nKepler results (manoeuvre epochs only):")
    flagged = df[df['likely_maneuver']]
    if flagged.empty:
        print("  No manoeuvres detected.")
    else:
        print(flagged[['epoch_from', 'epoch_to', 'gap_days',
                        'delta_sma', 'excess_delta_sma',
                        'delta_ecc', 'kp',
                        'confirmed_maneuver', 'uncertain_maneuver']]
              .to_string(index=False))

    return df


def run_od(args: argparse.Namespace, username: str, password: str):
    from orbit_analysis_od import analyse_maneuvers_od

    print("\n" + "═" * 60)
    print(f"  OD Analysis — CATNR {args.catnr}")
    print("═" * 60)

    df = analyse_maneuvers_od(
        catnr                 = args.catnr,
        username              = username,
        password              = password,
        data_dir              = args.data_dir,
        output_dir            = args.output_dir,
        sigma_multiplier      = args.sigma_multiplier,
        kp_threshold          = args.kp_threshold,
        max_gap_days          = args.max_gap_days,
        use_bstar             = args.use_bstar,
    )

    print("\nOD results (manoeuvre epochs only):")
    flagged = df[df['likely_maneuver']]
    if flagged.empty:
        print("  No manoeuvres detected.")
    else:
        print(flagged[['epoch_from', 'epoch_to', 'gap_days',
                        'vel_residual_ms', 'delta_sma', 'delta_ecc',
                        'bstar', 'delta_bstar', 'kp',
                        'od_flag', 'bstar_flag',
                        'confirmed_maneuver', 'uncertain_maneuver']]
              .to_string(index=False))

    return df


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    os.makedirs(args.data_dir,   exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Satellite:  CATNR {args.catnr}")
    print(f"Method:     {args.method}")
    print(f"Data dir:   {args.data_dir}")
    print(f"Output dir: {args.output_dir}")

    # Resolve OD credentials up front if needed (avoids mid-run prompt)
    username = password = None
    if args.method in ("od", "both"):
        username, password = resolve_credentials()

    results = {}

    if args.method in ("kepler", "both"):
        try:
            results["kepler"] = run_kepler(args)
        except Exception as exc:
            print(f"\nERROR: Kepler analysis failed — {exc}", file=sys.stderr)
            if args.method == "kepler":
                sys.exit(1)

    if args.method in ("od", "both"):
        try:
            results["od"] = run_od(args, username, password)
        except Exception as exc:
            print(f"\nERROR: OD analysis failed — {exc}", file=sys.stderr)
            if args.method == "od":
                sys.exit(1)

    # ── Cross-method summary (only when both ran successfully) ────────────────
    if "kepler" in results and "od" in results:
        kep_confirmed = results["kepler"]['confirmed_maneuver'].sum()
        kep_uncertain = results["kepler"]['uncertain_maneuver'].sum()
        od_confirmed  = results["od"]['confirmed_maneuver'].sum()
        od_uncertain  = results["od"]['uncertain_maneuver'].sum()

        print("\n" + "═" * 60)
        print(f"  Cross-Method Summary — CATNR {args.catnr}")
        print("═" * 60)
        print(f"  {'Method':<10}  {'Confirmed':>10}  {'Uncertain':>10}")
        print(f"  {'-'*10}  {'-'*10}  {'-'*10}")
        print(f"  {'Kepler':<10}  {kep_confirmed:>10}  {kep_uncertain:>10}")
        print(f"  {'OD':<10}  {od_confirmed:>10}  {od_uncertain:>10}")
        print("═" * 60)
        print(f"  Plots saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
