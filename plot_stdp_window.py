#!/usr/bin/env python3
"""Simulate the RRAM STDP window versus delta-t and initial gap with Spectre.

The script is independent of ``main_stdp.py``.  It builds one isolated RRAM
pair for every requested initial-gap/timing combination and uses the same
Verilog-A waveform modules as the training circuit, with a reference-style
balanced ramp preset for this characterization:

* ``adaptive_spikegen.va`` drives the pre-synaptic RRAM terminal.
* ``spikegen.va`` drives the post-synaptic RRAM terminal.
* ``rram.va`` supplies the device dynamics and symmetric learning-rate scale.

Delta-t is defined as ``t_post - t_pre`` at the falling threshold crossings
that start the programming ramps. Thus positive delta-t means pre before post.
``--positive-dt-dg-ratio`` reduces the RRAM RESET state-update term, which is
the conductance-decrease branch produced by strictly positive delta-t. A ratio
of 0.4 therefore targets a positive-branch |delta-G| around 40% of the
negative branch while preserving physical RRAM simulation.
The default result is the baseline-corrected final conductance change: a 2D
depth map beside one delta-G timing curve per initial gap. Use
``--plot-metric resistance`` for delta-R, while ``--base-gap`` selects one gap
and produces a single timing curve.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np

from import_data import import_data


DEFAULT_DELTA_MIN_US = -4.0
DEFAULT_DELTA_MAX_US = 4.0
DEFAULT_DELTA_STEP_US = 0.25
DEFAULT_PRE_HIGH = 1.2
DEFAULT_POST_HIGH = 1.0
DEFAULT_PRE_THRESHOLD = 0.8
DEFAULT_POST_THRESHOLD = 0.8
DEFAULT_PRE_EDGE_US = 0.1
DEFAULT_POST_EDGE_US = 0.001
DEFAULT_PULSE_WIDTH_US = 1.0
DEFAULT_RAMP_US = 3.0
DEFAULT_ADAPTIVE_A_REF = -0.422
DEFAULT_ADAPTIVE_R_REF = 263244.0
DEFAULT_ADAPTIVE_A_MIN_MAG = 0.01
DEFAULT_ADAPTIVE_A_MAX_MAG = 0.50
DEFAULT_SET_DG_EQUALIZATION_EXPONENT = 1.61
DEFAULT_RESET_DG_EQUALIZATION_EXPONENT = 2.02
DEFAULT_DG_EQUALIZATION_CURVATURE = 0.30
DEFAULT_POST_A = -0.20
DEFAULT_GAP_MIN_NM = 0.90
DEFAULT_GAP_MAX_NM = 1.10
DEFAULT_GAP_STEP_NM = 0.025
DEFAULT_TIMESTEP_US = 0.2
DEFAULT_LEARNING_RATE_SCALE = 0.20
DEFAULT_POSITIVE_DT_DG_RATIO = 0.40
DEFAULT_THREADS = 4


def inclusive_values(
    minimum: float,
    maximum: float,
    step: float,
    option_name: str,
) -> np.ndarray:
    """Return an inclusive sweep after checking exact step divisibility."""
    if step <= 0.0:
        raise ValueError(f"{option_name} step must be positive")
    if maximum < minimum:
        raise ValueError(f"{option_name} maximum must be at least its minimum")
    intervals_exact = (maximum - minimum) / step
    intervals = int(round(intervals_exact))
    if not math.isclose(intervals_exact, intervals, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(
            f"{option_name} range must be exactly divisible by its step"
        )
    return np.linspace(minimum, maximum, intervals + 1, dtype=float)


def delta_values(minimum: float, maximum: float, step: float) -> np.ndarray:
    return inclusive_values(minimum, maximum, step, "delta-t")


def gap_values_nm(args: argparse.Namespace) -> np.ndarray:
    """Return initial gaps in nm; --base-gap requests legacy single-gap mode."""
    if args.base_gap is not None:
        gap_nm = args.base_gap * 1.0e9
        if not 0.9 <= gap_nm <= 1.1:
            raise ValueError("--base-gap must be between 0.9e-9 and 1.1e-9 m")
        return np.asarray([gap_nm], dtype=float)
    gaps_nm = inclusive_values(
        args.gap_min_nm,
        args.gap_max_nm,
        args.gap_step_nm,
        "initial-gap",
    )
    if gaps_nm[0] < 0.9 or gaps_nm[-1] > 1.1:
        raise ValueError("initial-gap sweep must stay between 0.9 and 1.1 nm")
    return gaps_nm


def pulse_points_us(
    falling_cross_us: float,
    high: float,
    threshold: float,
    edge_us: float,
    pulse_width_us: float,
    stop_us: float,
) -> list[tuple[float, float]]:
    """Make a PWL pulse whose threshold crossings have the requested timing."""
    rising_cross_us = falling_cross_us - pulse_width_us
    rise_start_us = rising_cross_us - edge_us * threshold / high
    rise_end_us = rise_start_us + edge_us
    fall_start_us = falling_cross_us - edge_us * (high - threshold) / high
    fall_end_us = fall_start_us + edge_us
    if rise_start_us <= 0.0:
        raise ValueError("internal pulse placement reached time zero")
    if rise_end_us >= fall_start_us:
        raise ValueError(
            "pulse width is too short for the selected voltage-edge duration"
        )
    return [
        (0.0, 0.0),
        (rise_start_us, 0.0),
        (rise_end_us, high),
        (fall_start_us, high),
        (fall_end_us, 0.0),
        (stop_us, 0.0),
    ]


def spectre_wave(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{time_us:.12g}u {voltage:.12g}" for time_us, voltage in points)


def write_netlist(
    path: Path,
    root: Path,
    deltas_us: np.ndarray,
    gaps_nm: np.ndarray,
    args: argparse.Namespace,
) -> tuple[float, list[str]]:
    """Write all gap/delta combinations and return stop time and signal order."""
    earliest_offset_us = min(0.0, float(deltas_us[0]))
    latest_offset_us = max(0.0, float(deltas_us[-1]))
    edge_margin_us = 2.0 * max(args.pre_edge_us, args.post_edge_us)
    pre_cross_us = (
        args.settle_before_us
        + args.pulse_width_us
        + edge_margin_us
        - earliest_offset_us
    )
    latest_cross_us = pre_cross_us + latest_offset_us
    stop_us = latest_cross_us + max(args.pre_ramp_us, args.post_ramp_us)
    stop_us += args.settle_after_us + edge_margin_us

    signals: list[str] = []
    for gap_index in range(len(gaps_nm)):
        signals.append(f"r_base_g{gap_index:03d}")
        signals.extend(
            f"r_out_g{gap_index:03d}_d{delta_index:04d}"
            for delta_index in range(len(deltas_us))
        )

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("// STDP initial-gap and timing-window sweep\n")
        handle.write("// delta_t = t_post - t_pre at falling threshold crossings\n")
        handle.write(
            "// positive-delta-t delta-G ratio = "
            f"{args.positive_dt_dg_ratio:.12g}\n"
        )
        handle.write("simulator lang=spectre\n\n")
        handle.write(f'ahdl_include "{root / "rram.va"}"\n')
        handle.write(f'ahdl_include "{root / "spikegen.va"}"\n')
        handle.write(f'ahdl_include "{root / "adaptive_spikegen.va"}"\n\n')

        # One pre/post source pair is shared across all initial gaps for the
        # same delta-t. Verilog-A inputs are high impedance.
        for delta_index, delta_us in enumerate(deltas_us):
            delta_suffix = f"d{delta_index:04d}"
            post_cross_us = pre_cross_us + float(delta_us)
            pre_points = pulse_points_us(
                pre_cross_us,
                args.pre_high,
                args.pre_threshold,
                args.pre_edge_us,
                args.pulse_width_us,
                stop_us,
            )
            post_points = pulse_points_us(
                post_cross_us,
                args.post_high,
                args.post_threshold,
                args.post_edge_us,
                args.pulse_width_us,
                stop_us,
            )
            handle.write(f"// delta_t={delta_us:.12g} us\n")
            handle.write(
                f"Vpre_{delta_suffix} (pre_in_{delta_suffix} 0) vsource type=pwl "
                f"wave=[{spectre_wave(pre_points)}]\n"
            )
            handle.write(
                f"Vpost_{delta_suffix} (post_in_{delta_suffix} 0) vsource type=pwl "
                f"wave=[{spectre_wave(post_points)}]\n"
            )
        handle.write("\n")

        for gap_index, gap_nm in enumerate(gaps_nm):
            gap_suffix = f"g{gap_index:03d}"
            gap_m = float(gap_nm) * 1.0e-9
            handle.write(f"// initial_gap={gap_nm:.12g} nm\n")
            # A zero-voltage reference per initial gap removes tiny drift that
            # is unrelated to the paired programming waveforms.
            handle.write(
                f"Xbase_{gap_suffix} (0 0 r_base_{gap_suffix}) RRAM "
                f"gap_ini={gap_m:.12g} tstep={args.timestep_us:.12g}u "
                f"learning_rate_scale={args.learning_rate_scale:.12g} "
                f"reset_learning_rate_ratio="
                f"{args.positive_dt_dg_ratio:.12g}\n"
            )

            for delta_index, _delta_us in enumerate(deltas_us):
                delta_suffix = f"d{delta_index:04d}"
                pair_suffix = f"{gap_suffix}_{delta_suffix}"
                r_out = f"r_out_{pair_suffix}"
                handle.write(
                    f"Xpre_{pair_suffix} (pre_in_{delta_suffix} {r_out} "
                    f"pre_rram_{pair_suffix}) adaptive_spikegen "
                    f"vth={args.pre_threshold:.12g} "
                    f"a_ref={args.adaptive_a_ref:.12g} b=0 "
                    f"Tr={args.pre_ramp_us:.12g}u "
                    f"r_ref={args.adaptive_r_ref:.12g} "
                    f"a_min_mag={args.adaptive_a_min_mag:.12g} "
                    f"a_max_mag={args.adaptive_a_max_mag:.12g} "
                    f"set_dg_equalization_exponent="
                    f"{args.set_dg_equalization_exponent:.12g} "
                    f"reset_dg_equalization_exponent="
                    f"{args.reset_dg_equalization_exponent:.12g} "
                    f"dg_equalization_curvature="
                    f"{args.dg_equalization_curvature:.12g}\n"
                )
                handle.write(
                    f"Xpost_{pair_suffix} (post_in_{delta_suffix} "
                    f"post_rram_{pair_suffix}) spikegen "
                    f"vth={args.post_threshold:.12g} a={args.post_a:.12g} b=0 "
                    f"Tr={args.post_ramp_us:.12g}u\n"
                )
                handle.write(
                    f"Xrram_{pair_suffix} (pre_rram_{pair_suffix} "
                    f"post_rram_{pair_suffix} {r_out}) RRAM "
                    f"gap_ini={gap_m:.12g} "
                    f"tstep={args.timestep_us:.12g}u "
                    f"learning_rate_scale={args.learning_rate_scale:.12g} "
                    f"reset_learning_rate_ratio="
                    f"{args.positive_dt_dg_ratio:.12g}\n"
                )
            handle.write("\n")

        handle.write("simOptions options save=selected currents=selected\n")
        for signal in signals:
            handle.write(f"save {signal}\n")
        handle.write(
            f"tran tran stop={stop_us:.12g}u "
            f"maxstep={args.timestep_us:.12g}u "
            f"strobeperiod={stop_us:.12g}u strobedelay=0 "
            "strobeoutput=strobeonly\n"
        )

    return stop_us, signals


def run_spectre(
    netlist: Path,
    raw_path: Path,
    log_path: Path,
    executable: str,
    aps: bool,
    threads: int,
) -> None:
    spectre = shutil.which(executable) if "/" not in executable else executable
    if not spectre:
        raise FileNotFoundError(
            f"Cannot find Spectre executable '{executable}'. Use --spectre PATH."
        )
    command = [str(spectre)]
    if aps:
        command.append("++aps")
    if threads > 0:
        command.append(f"+mt={threads}")
    command.extend(
        [
            str(netlist.resolve()),
            "-format",
            "nutascii",
            "-raw",
            str(raw_path.resolve()),
            "+log",
            str(log_path.resolve()),
        ]
    )
    print("Running:", " ".join(command))
    completed = subprocess.run(command, cwd=netlist.parent, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Spectre exited with status {completed.returncode}. See {log_path}."
        )


def calculate_window(
    raw_path: Path,
    signals: list[str],
    deltas_us: np.ndarray,
    gaps_nm: np.ndarray,
) -> tuple[np.ndarray, ...]:
    data, _ = import_data(raw_path, signals)
    resistances = data[:, 1:]
    if np.any(~np.isfinite(resistances)) or np.any(resistances <= 0.0):
        raise ValueError("Spectre produced invalid RRAM resistance values")
    row_width = 1 + len(deltas_us)
    expected_signals = len(gaps_nm) * row_width
    if resistances.shape[1] != expected_signals:
        raise ValueError(
            "raw result does not match the requested initial-gap/delta-t sweep"
        )
    initial_r_rows = resistances[0].reshape(len(gaps_nm), row_width)
    final_r_rows = resistances[-1].reshape(len(gaps_nm), row_width)
    baseline_delta_r = final_r_rows[:, 0] - initial_r_rows[:, 0]
    initial_r_ohm = initial_r_rows[:, 1:]
    final_r_ohm = final_r_rows[:, 1:]
    delta_r_ohm = final_r_ohm - initial_r_ohm - baseline_delta_r[:, None]
    relative_r_percent = 100.0 * delta_r_ohm / initial_r_ohm

    conductance_us = 1.0e6 / resistances
    initial_g_rows = conductance_us[0].reshape(len(gaps_nm), row_width)
    final_g_rows = conductance_us[-1].reshape(len(gaps_nm), row_width)
    baseline_delta_g = final_g_rows[:, 0] - initial_g_rows[:, 0]
    initial_g_us = initial_g_rows[:, 1:]
    final_g_us = final_g_rows[:, 1:]
    delta_g_us = final_g_us - initial_g_us - baseline_delta_g[:, None]
    relative_g_percent = 100.0 * delta_g_us / initial_g_us
    return (
        initial_r_ohm,
        final_r_ohm,
        delta_r_ohm,
        relative_r_percent,
        initial_g_us,
        final_g_us,
        delta_g_us,
        relative_g_percent,
    )


def write_results_csv(
    path: Path,
    deltas_us: np.ndarray,
    gaps_nm: np.ndarray,
    initial_r_ohm: np.ndarray,
    final_r_ohm: np.ndarray,
    delta_r_ohm: np.ndarray,
    relative_r_percent: np.ndarray,
    initial_g_us: np.ndarray,
    final_g_us: np.ndarray,
    delta_g_us: np.ndarray,
    relative_g_percent: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "initial_gap_nm",
                "delta_t_ns",
                "delta_t_us",
                "initial_resistance_ohm",
                "final_resistance_ohm",
                "delta_resistance_ohm",
                "relative_delta_resistance_percent",
                "initial_conductance_us",
                "final_conductance_us",
                "delta_conductance_us",
                "relative_delta_conductance_percent",
            ]
        )
        for gap_index, gap_nm in enumerate(gaps_nm):
            for delta_index, delta_us in enumerate(deltas_us):
                writer.writerow(
                    [
                        gap_nm,
                        delta_us * 1000.0,
                        delta_us,
                        initial_r_ohm[gap_index, delta_index],
                        final_r_ohm[gap_index, delta_index],
                        delta_r_ohm[gap_index, delta_index],
                        relative_r_percent[gap_index, delta_index],
                        initial_g_us[gap_index, delta_index],
                        final_g_us[gap_index, delta_index],
                        delta_g_us[gap_index, delta_index],
                        relative_g_percent[gap_index, delta_index],
                    ]
                )


def plot_window(
    path: Path,
    deltas_us: np.ndarray,
    gaps_nm: np.ndarray,
    delta_r_ohm: np.ndarray,
    delta_g_us: np.ndarray,
    plot_metric: str,
    learning_rate_scale: float,
    positive_dt_dg_ratio: float,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm

    path.parent.mkdir(parents=True, exist_ok=True)
    delta_ns = deltas_us * 1000.0
    if plot_metric == "resistance":
        values = delta_r_ohm
        symbol = "ΔR"
        unit = "Ω"
        quantity = "Resistance"
    else:
        values = delta_g_us
        symbol = "ΔG"
        unit = "µS"
        quantity = "Conductance"

    if len(gaps_nm) == 1:
        figure, axis = plt.subplots(figsize=(8.0, 5.2))
        axis.axhline(0.0, color="0.35", linewidth=0.9)
        axis.axvline(0.0, color="0.35", linewidth=0.9, linestyle="--")
        axis.plot(delta_ns, values[0], color="tab:blue", marker="o", markersize=3.5)
        axis.set_xlabel("Gap Time Δt = t_post − t_pre (ns)")
        axis.set_ylabel(f"{symbol} ({unit})")
        axis.set_title(
            f"{symbol} vs Gap Time, gap_ini = {gaps_nm[0]:.2f} nm\n"
            f"Learning-rate scale = {learning_rate_scale:g}, "
            f"positive-Δt ratio = {positive_dt_dg_ratio:g}"
        )
        axis.grid(True, alpha=0.3)
    else:
        def color_norm(values: np.ndarray) -> Normalize:
            minimum = float(np.nanmin(values))
            maximum = float(np.nanmax(values))
            if minimum < 0.0 < maximum:
                return TwoSlopeNorm(vmin=minimum, vcenter=0.0, vmax=maximum)
            if maximum <= minimum:
                maximum = minimum + max(abs(minimum) * 1.0e-9, 1.0e-15)
            return Normalize(vmin=minimum, vmax=maximum)

        figure, axes = plt.subplots(1, 2, figsize=(14.0, 5.3))
        depth_map = axes[0].pcolormesh(
            delta_ns,
            gaps_nm,
            values,
            shading="auto",
            cmap="viridis",
            norm=color_norm(values),
        )
        figure.colorbar(depth_map, ax=axes[0], label=f"{symbol} ({unit})")
        axes[0].set_title(f"{symbol} Final")
        axes[0].set_xlabel("Gap Time (ns)")
        axes[0].set_ylabel("gap_ini (nm)")
        axes[0].axvline(0.0, color="black", linewidth=0.8, linestyle="--")

        colors = plt.cm.rainbow(np.linspace(0.0, 1.0, len(gaps_nm)))
        for gap_index in range(len(gaps_nm) - 1, -1, -1):
            axes[1].plot(
                delta_ns,
                values[gap_index],
                color=colors[gap_index],
                linewidth=1.5,
                label=f"{gaps_nm[gap_index]:.2f} nm",
            )
        axes[1].set_title(f"{symbol} vs Gap Time")
        axes[1].set_xlabel("Gap Time (ns)")
        axes[1].set_ylabel(f"{symbol} ({unit})")
        axes[1].axhline(0.0, color="0.35", linewidth=0.8)
        axes[1].axvline(0.0, color="black", linewidth=0.8, linestyle="--")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(loc="best", fontsize=8, ncol=1)
        figure.suptitle(
            f"RRAM STDP {quantity}-Change Window\n"
            f"Learning-rate scale = {learning_rate_scale:g}, "
            f"positive-Δt ratio = {positive_dt_dg_ratio:g}"
        )

    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    print(f"Saved {path}")
    if show:
        plt.show()
    else:
        plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta-min-us", type=float, default=DEFAULT_DELTA_MIN_US)
    parser.add_argument("--delta-max-us", type=float, default=DEFAULT_DELTA_MAX_US)
    parser.add_argument("--delta-step-us", type=float, default=DEFAULT_DELTA_STEP_US)
    parser.add_argument("--pulse-width-us", type=float, default=DEFAULT_PULSE_WIDTH_US)
    parser.add_argument("--pre-high", type=float, default=DEFAULT_PRE_HIGH)
    parser.add_argument("--post-high", type=float, default=DEFAULT_POST_HIGH)
    parser.add_argument("--pre-threshold", type=float, default=DEFAULT_PRE_THRESHOLD)
    parser.add_argument("--post-threshold", type=float, default=DEFAULT_POST_THRESHOLD)
    parser.add_argument("--pre-edge-us", type=float, default=DEFAULT_PRE_EDGE_US)
    parser.add_argument("--post-edge-us", type=float, default=DEFAULT_POST_EDGE_US)
    parser.add_argument("--pre-ramp-us", type=float, default=DEFAULT_RAMP_US)
    parser.add_argument("--post-ramp-us", type=float, default=DEFAULT_RAMP_US)
    parser.add_argument(
        "--adaptive-a-ref",
        type=float,
        default=DEFAULT_ADAPTIVE_A_REF,
        help=(
            "pre-synaptic adaptive ramp reference voltage; the balanced "
            f"plot default is {DEFAULT_ADAPTIVE_A_REF:g} V"
        ),
    )
    parser.add_argument(
        "--adaptive-r-ref",
        type=float,
        default=DEFAULT_ADAPTIVE_R_REF,
        help=(
            "resistance reference for the adaptive pre-ramp; the reference "
            f"plot preset is {DEFAULT_ADAPTIVE_R_REF:g} ohm"
        ),
    )
    parser.add_argument(
        "--adaptive-a-min-mag", type=float, default=DEFAULT_ADAPTIVE_A_MIN_MAG
    )
    parser.add_argument(
        "--adaptive-a-max-mag",
        type=float,
        default=DEFAULT_ADAPTIVE_A_MAX_MAG,
        help=(
            "maximum pre-synaptic ramp magnitude; the balanced plot default "
            f"is {DEFAULT_ADAPTIVE_A_MAX_MAG:g} V"
        ),
    )
    parser.add_argument(
        "--set-dg-equalization-exponent",
        type=float,
        default=DEFAULT_SET_DG_EQUALIZATION_EXPONENT,
        help="finite-pulse correction exponent for conductance increase",
    )
    parser.add_argument(
        "--reset-dg-equalization-exponent",
        type=float,
        default=DEFAULT_RESET_DG_EQUALIZATION_EXPONENT,
        help="finite-pulse correction exponent for conductance decrease",
    )
    parser.add_argument(
        "--dg-equalization-curvature",
        type=float,
        default=DEFAULT_DG_EQUALIZATION_CURVATURE,
        help="symmetric finite-pulse correction in log-resistance space",
    )
    parser.add_argument("--post-a", type=float, default=DEFAULT_POST_A)
    parser.add_argument(
        "--base-gap",
        type=float,
        default=None,
        help=(
            "simulate one initial gap in meters and generate the legacy line "
            "plot; overrides --gap-min-nm/--gap-max-nm/--gap-step-nm"
        ),
    )
    parser.add_argument(
        "--gap-min-nm",
        type=float,
        default=DEFAULT_GAP_MIN_NM,
        help=f"minimum initial gap in nm (default: {DEFAULT_GAP_MIN_NM:g})",
    )
    parser.add_argument(
        "--gap-max-nm",
        type=float,
        default=DEFAULT_GAP_MAX_NM,
        help=f"maximum initial gap in nm (default: {DEFAULT_GAP_MAX_NM:g})",
    )
    parser.add_argument(
        "--gap-step-nm",
        type=float,
        default=DEFAULT_GAP_STEP_NM,
        help=f"initial-gap step in nm (default: {DEFAULT_GAP_STEP_NM:g})",
    )
    parser.add_argument("--timestep-us", type=float, default=DEFAULT_TIMESTEP_US)
    parser.add_argument(
        "--plot-metric",
        choices=("resistance", "conductance"),
        default="conductance",
        help="quantity shown in the figure (default: conductance)",
    )
    parser.add_argument(
        "--learning-rate-scale",
        type=float,
        default=DEFAULT_LEARNING_RATE_SCALE,
        help=(
            "symmetric RRAM state-update multiplier for SET and RESET "
            f"(default: {DEFAULT_LEARNING_RATE_SCALE:g})"
        ),
    )
    parser.add_argument(
        "--positive-dt-dg-ratio",
        type=float,
        default=DEFAULT_POSITIVE_DT_DG_RATIO,
        help=(
            "RRAM RESET/update-rate ratio for the t_post - t_pre > 0 "
            "conductance-decrease branch; 0 disables RESET and 1 preserves "
            "the symmetric window "
            f"(default: {DEFAULT_POSITIVE_DT_DG_RATIO:g})"
        ),
    )
    parser.add_argument("--settle-before-us", type=float, default=1.0)
    parser.add_argument("--settle-after-us", type=float, default=1.0)
    parser.add_argument("--spectre", default="spectre")
    parser.add_argument("--workdir", type=Path, default=root / "stdp_window")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument(
        "--aps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable Spectre APS acceleration (default: enabled)",
    )
    parser.add_argument("--no-run", action="store_true", help="only generate the sweep netlist")
    parser.add_argument("--no-show", action="store_true", help="save without opening a window")
    parser.add_argument("--figure", type=Path, help="output PNG (default: WORKDIR/stdp_window.png)")
    parser.add_argument("--csv", type=Path, help="output CSV (default: WORKDIR/stdp_window.csv)")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.threads < 0:
        raise ValueError("--threads must be zero or positive")
    if args.pulse_width_us <= 0.0:
        raise ValueError("--pulse-width-us must be positive")
    if args.pre_edge_us <= 0.0 or args.post_edge_us <= 0.0:
        raise ValueError("edge durations must be positive")
    if args.pre_ramp_us <= 0.0 or args.post_ramp_us <= 0.0:
        raise ValueError("ramp durations must be positive")
    if not 0.0 < args.pre_threshold < args.pre_high:
        raise ValueError("--pre-threshold must be between zero and --pre-high")
    if not 0.0 < args.post_threshold < args.post_high:
        raise ValueError("--post-threshold must be between zero and --post-high")
    if args.adaptive_r_ref <= 0.0:
        raise ValueError("--adaptive-r-ref must be positive")
    if args.adaptive_a_min_mag < 0.0:
        raise ValueError("--adaptive-a-min-mag must be zero or positive")
    if args.adaptive_a_max_mag <= 0.0:
        raise ValueError("--adaptive-a-max-mag must be positive")
    if args.adaptive_a_min_mag > args.adaptive_a_max_mag:
        raise ValueError("adaptive minimum magnitude cannot exceed maximum")
    if args.set_dg_equalization_exponent <= 0.0:
        raise ValueError("--set-dg-equalization-exponent must be positive")
    if args.reset_dg_equalization_exponent <= 0.0:
        raise ValueError("--reset-dg-equalization-exponent must be positive")
    if args.dg_equalization_curvature < 0.0:
        raise ValueError("--dg-equalization-curvature cannot be negative")
    if args.timestep_us <= 0.0:
        raise ValueError("--timestep-us must be positive")
    if args.learning_rate_scale < 0.0:
        raise ValueError("--learning-rate-scale must be zero or positive")
    if not 0.0 <= args.positive_dt_dg_ratio <= 1.0:
        raise ValueError("--positive-dt-dg-ratio must be between 0 and 1")
    if args.settle_before_us < 0.0 or args.settle_after_us < 0.0:
        raise ValueError("settling durations cannot be negative")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    deltas_us = delta_values(args.delta_min_us, args.delta_max_us, args.delta_step_us)
    gaps_nm = gap_values_nm(args)
    root = Path(__file__).resolve().parent
    workdir = args.workdir.expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    netlist = workdir / "stdp_window.scs"
    raw_path = workdir / "stdp_window.raw"
    log_path = workdir / "spectre.log"
    figure_path = (args.figure or workdir / "stdp_window.png").expanduser().resolve()
    csv_path = (args.csv or workdir / "stdp_window.csv").expanduser().resolve()

    stop_us, signals = write_netlist(netlist, root, deltas_us, gaps_nm, args)
    print(
        f"Generated {netlist}: {len(gaps_nm)} initial gaps from "
        f"{gaps_nm[0]:g} to {gaps_nm[-1]:g} nm, {len(deltas_us)} delta-t "
        f"values from {deltas_us[0]:g} to {deltas_us[-1]:g} us "
        f"({len(gaps_nm) * len(deltas_us)} combinations); stop={stop_us:g} us; "
        f"positive-delta-t delta-G ratio={args.positive_dt_dg_ratio:g}"
    )
    if args.no_run:
        return

    run_spectre(netlist, raw_path, log_path, args.spectre, args.aps, args.threads)
    (
        initial_r_ohm,
        final_r_ohm,
        delta_r_ohm,
        relative_r_percent,
        initial_g_us,
        final_g_us,
        delta_g_us,
        relative_g_percent,
    ) = calculate_window(raw_path, signals, deltas_us, gaps_nm)
    write_results_csv(
        csv_path,
        deltas_us,
        gaps_nm,
        initial_r_ohm,
        final_r_ohm,
        delta_r_ohm,
        relative_r_percent,
        initial_g_us,
        final_g_us,
        delta_g_us,
        relative_g_percent,
    )
    print(f"Saved {csv_path}")
    plot_window(
        figure_path,
        deltas_us,
        gaps_nm,
        delta_r_ohm,
        delta_g_us,
        args.plot_metric,
        args.learning_rate_scale,
        args.positive_dt_dg_ratio,
        not args.no_show,
    )

    if len(gaps_nm) > 1:
        mean_delta_g = np.mean(delta_g_us, axis=0)
        set_index = int(np.argmax(mean_delta_g))
        reset_index = int(np.argmin(mean_delta_g))
        for branch, delta_index in (("SET", set_index), ("RESET", reset_index)):
            magnitudes = np.abs(delta_g_us[:, delta_index])
            mean_magnitude = float(np.mean(magnitudes))
            span_percent = (
                100.0 * float(np.max(magnitudes) - np.min(magnitudes))
                / mean_magnitude
                if mean_magnitude > 0.0
                else 0.0
            )
            print(
                f"{branch} gap-equality at delta-t={deltas_us[delta_index]:g} us: "
                f"mean |delta-G|={mean_magnitude:.6g} uS, "
                f"max-min span={span_percent:.3f}%"
            )

    if args.plot_metric == "resistance":
        reported_values = delta_r_ohm
        reported_name = "resistance"
        reported_unit = "ohm"
    else:
        reported_values = delta_g_us
        reported_name = "conductance"
        reported_unit = "uS"
    maximum_index = np.unravel_index(
        int(np.argmax(reported_values)), reported_values.shape
    )
    minimum_index = np.unravel_index(
        int(np.argmin(reported_values)), reported_values.shape
    )
    maximum_gap_index, maximum_delta_index = maximum_index
    minimum_gap_index, minimum_delta_index = minimum_index
    print(
        f"Largest {reported_name} increase: "
        f"{reported_values[maximum_index]:.6g} {reported_unit} at delta-t="
        f"{deltas_us[maximum_delta_index]:g} us, initial gap="
        f"{gaps_nm[maximum_gap_index]:g} nm"
    )
    if reported_values[minimum_index] < 0.0:
        print(
            f"Largest {reported_name} decrease: "
            f"{reported_values[minimum_index]:.6g} {reported_unit} at delta-t="
            f"{deltas_us[minimum_delta_index]:g} us, initial gap="
            f"{gaps_nm[minimum_gap_index]:g} nm"
        )
    else:
        print(
            f"No {reported_name} decrease occurred; minimum change was "
            f"{reported_values[minimum_index]:.6g} {reported_unit} at delta-t="
            f"{deltas_us[minimum_delta_index]:g} us, initial gap="
            f"{gaps_nm[minimum_gap_index]:g} nm"
        )


if __name__ == "__main__":
    main()
