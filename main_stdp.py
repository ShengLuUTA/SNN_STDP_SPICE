#!/usr/bin/env python3
"""Generate, run, and plot the SNN/RRAM STDP circuit with Cadence Spectre.

The topology and naming follow ``main_stdp.m``.  The generated ``stdp.sp``
uses the SPICE-compatible Spectre format demonstrated by
``spectre_test/single_neuron.sp``: the circuit stays in
``simulator lang=spice``, while native-only statements are enclosed by an
explicit ``simulator lang=spectre`` block.
"""

from __future__ import annotations

import argparse
import math
import random
import re
import shutil
import subprocess
from pathlib import Path
from typing import TextIO

DEFAULT_N = 64
DEFAULT_M = 4
DEFAULT_TIMESTEP_US = 0.2
DEFAULT_SIMULATION_US = 100.0
DEFAULT_SAVE_PERIOD_US = None
DEFAULT_THREADS = 8
DEFAULT_INPUT_PWL = "mnist_digit1_20samples_7.sp"
DEFAULT_ADAPTIVE_A_REF = -0.422
DEFAULT_ADAPTIVE_R_REF = 263244.0
DEFAULT_ADAPTIVE_A_MIN_MAG = 0.01
DEFAULT_ADAPTIVE_A_MAX_MAG = 0.50
DEFAULT_SET_DG_EQUALIZATION_EXPONENT = 1.61
DEFAULT_RESET_DG_EQUALIZATION_EXPONENT = 2.02
DEFAULT_DG_EQUALIZATION_CURVATURE = 0.30
DEFAULT_LEARNING_RATE_SCALE = 0.20
DEFAULT_POSITIVE_DT_DG_RATIO = 0.40
DEFAULT_MEM_THRESHOLD = 0.40
RRAM_GAP_MIN_M = 0.9e-9
RRAM_GAP_MAX_M = 1.1e-9

_PWL_SOURCE_RE = re.compile(
    r"^\s*Vin(?P<source>\d+)\s+vin(?P<node>\d+)\s+0\s+PWL\s*\(",
    re.IGNORECASE,
)
_TOTAL_DURATION_RE = re.compile(
    r"\btotal\s*=\s*(?P<value>[0-9.eE+-]+)\s*(?P<unit>ms|us|ns|ps|s)\b",
    re.IGNORECASE,
)
_DURATION_TO_US = {"s": 1.0e6, "ms": 1.0e3, "us": 1.0, "ns": 1.0e-3, "ps": 1.0e-6}


def expected_signals(n_inputs: int, n_neurons: int) -> list[str]:
    signals: list[str] = []
    for neuron in range(1, n_neurons + 1):
        signals.extend([f"vout_SRC{neuron}", f"node2_{neuron}"])
        signals.extend(
            f"XNEU{neuron}.R_out{rram}" for rram in range(1, n_inputs + 1)
        )
    return signals


def inspect_input_pwl(path: str | Path, n_inputs: int) -> tuple[int, float | None]:
    """Validate the encoder PWL deck and return source count and duration in us."""
    input_path = Path(path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Encoded input PWL file not found: {input_path}. "
            "Run main_encoding.py first or pass --input-pwl PATH."
        )

    source_ids: set[int] = set()
    duration_us: float | None = None
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            source_match = _PWL_SOURCE_RE.match(line)
            if source_match:
                source_id = int(source_match.group("source"))
                node_id = int(source_match.group("node"))
                if source_id != node_id:
                    raise ValueError(
                        f"PWL source/node mismatch in {input_path}: "
                        f"Vin{source_id} drives vin{node_id}"
                    )
                source_ids.add(source_id)

            duration_match = _TOTAL_DURATION_RE.search(line)
            if duration_match:
                unit = duration_match.group("unit").casefold()
                duration_us = float(duration_match.group("value")) * _DURATION_TO_US[unit]

    required_ids = set(range(1, n_inputs + 1))
    missing_ids = sorted(required_ids - source_ids)
    if missing_ids:
        preview = ", ".join(map(str, missing_ids[:8]))
        if len(missing_ids) > 8:
            preview += ", ..."
        raise ValueError(
            f"{input_path} does not provide all {n_inputs} requested inputs; "
            f"missing Vin IDs: {preview}"
        )
    if not source_ids:
        raise ValueError(f"No Vin# PWL sources found in {input_path}")
    return len(source_ids), duration_us


def _write_header(handle: TextIO, root: Path, input_pwl: Path) -> None:
    root = root.resolve()
    input_pwl = input_pwl.expanduser().resolve()
    handle.write("******************************************************\n")
    handle.write("* SNN + RRAM STDP\n")
    handle.write("* Spectre SPICE Compatible Version\n")
    handle.write("******************************************************\n\n")
    handle.write("simulator lang=spice\n\n")
    handle.write(".title SNN + RRAM STDP (Gaussian gap variation + lateral connections)\n\n")
    handle.write("* === Simulation Options ===\n")
    handle.write(".option ingold=1\n\n")
    handle.write("* === SPICE Libraries ===\n")
    handle.write(f'.include "{root / "TLV2372.LIB"}"\n')
    handle.write(f'.include "{root / "cmp.sp"}"\n')
    handle.write(f'.include "{input_pwl}"\n')
    handle.write(f'.LIB "{root / "sm046005-1j.hspice"}" typical\n')
    handle.write("\n* === Native Spectre Verilog-A Includes ===\n")
    handle.write("simulator lang=spectre\n")
    for model in (
        "edge_to_pulse.va",
        "rram.va",
        "spikegen.va",
        "adaptive_spikegen.va",
        "follow_res.va",
    ):
        handle.write(f'ahdl_include "{root / model}"\n')
    handle.write("simulator lang=spice\n\n")


def _write_power(handle: TextIO, mem_threshold: float) -> None:
    handle.write("* === Power-Supply Parameters ===\n")
    handle.write(".param VDD_VAL=1.2\n")
    handle.write(".param VEE_VAL=-1.2\n")
    handle.write(".param VDDL_VAL=1.2\n")
    handle.write(f".param VDDC_VAL={mem_threshold:.12g}\n\n")
    handle.write("* === Power Supplies ===\n")
    handle.write("VDD  vdd  0 DC VDD_VAL\n")
    handle.write("VEE  vee  0 DC VEE_VAL\n")
    handle.write("VDDL vddl 0 DC VDDL_VAL\n")
    handle.write("VDDC vddc 0 DC VDDC_VAL\n\n")


def _write_neurons(
    handle: TextIO,
    n_inputs: int,
    n_neurons: int,
    base_gap: float,
    gap_sigma: float,
    rng: random.Random,
) -> None:
    handle.write("* === M Neuron Instances ===\n")
    input_nodes = " ".join(f"vin{i}" for i in range(1, n_inputs + 1))
    for neuron in range(1, n_neurons + 1):
        sampled_gap = base_gap * (1.0 + gap_sigma * rng.gauss(0.0, 1.0))
        gap_value = min(max(sampled_gap, RRAM_GAP_MIN_M), RRAM_GAP_MAX_M)
        handle.write(
            f"XNEU{neuron} {input_nodes} vdd vee vddl vddc "
            f"vout_SRC{neuron} node2_{neuron} single_neuron "
            f"gap_ini_base={gap_value:.12g}\n"
        )
    handle.write("\n")


def _write_lateral_connections(handle: TextIO, n_neurons: int) -> None:
    handle.write("* === Lateral Connections: vout_SRC_m -> node2_k (k != m) ===\n")
    for neuron in range(1, n_neurons + 1):
        for source in range(1, n_neurons + 1):
            if source != neuron:
                handle.write(
                    f"M_{neuron}_{source} node2_{neuron} vout_SRC{source} "
                    f"XNEU{neuron}.node1 0 nmos_3p3 L=0.35u W=20u\n"
                )
    handle.write("\n")


def _write_initial_conditions(handle: TextIO, n_neurons: int) -> None:
    handle.write("* === Initial Conditions ===\n")
    internal_nodes = ("vout", "node1", "node3", "node4", "node6", "node7", "node8")
    values: list[str] = []
    for neuron in range(1, n_neurons + 1):
        values.extend([f"V(vout_SRC{neuron})=0", f"V(node2_{neuron})=0"])
        values.extend(f"V(XNEU{neuron}.{node})=0" for node in internal_nodes)
    handle.write(".ic \\\n")
    for index, value in enumerate(values):
        suffix = " \\\n" if index < len(values) - 1 else "\n"
        handle.write(f"+ {value}{suffix}")
    handle.write("\n")


def _write_single_neuron(
    handle: TextIO,
    n_inputs: int,
    timestep_us: float,
    adaptive_a_ref: float,
    adaptive_r_ref: float,
    adaptive_a_min_mag: float,
    adaptive_a_max_mag: float,
    set_dg_equalization_exponent: float,
    reset_dg_equalization_exponent: float,
    dg_equalization_curvature: float,
    learning_rate_scale: float,
    positive_dt_dg_ratio: float,
) -> None:
    input_nodes = " ".join(f"vin{i}" for i in range(1, n_inputs + 1))
    handle.write(
        f".subckt single_neuron {input_nodes} vdd vee vddl vddc vout_SRC node2 "
        "gap_ini_base=1e-9\n"
    )
    # In Spectre's SPICE mode, parameters are assignments at the end of the
    # .subckt line. PSpice's ``PARAMS:`` marker is treated as an extra terminal.
    handle.write("* --- Input channels ---\n")
    for index in range(1, n_inputs + 1):
        handle.write(f"EIN{index}  TOP{index}  0 vin{index} 0 1\n")
        handle.write(f"EINV{index} vinc{index} 0 vin{index} 0 -1\n")
        handle.write(f"Xrw{index}  vinc{index} node1 R_out{index} follow_res\n")
        handle.write(
            f"Xspk{index} TOP{index} R_out{index} TOP_SRC{index} "
            "adaptive_spikegen vth=0.8 "
            f"a_ref={adaptive_a_ref:.12g} b=0.0 Tr=3u "
            f"r_ref={adaptive_r_ref:.12g} "
            f"a_min_mag={adaptive_a_min_mag:.12g} "
            f"a_max_mag={adaptive_a_max_mag:.12g} "
            f"set_dg_equalization_exponent="
            f"{set_dg_equalization_exponent:.12g} "
            f"reset_dg_equalization_exponent="
            f"{reset_dg_equalization_exponent:.12g} "
            f"dg_equalization_curvature={dg_equalization_curvature:.12g}\n"
        )
        handle.write(
            f"Xrram{index} TOP_SRC{index} vout_SRC R_out{index} rram "
            f"gap_ini=gap_ini_base tstep={timestep_us:.12g}u "
            f"learning_rate_scale={learning_rate_scale:.12g} "
            f"reset_learning_rate_ratio={positive_dt_dg_ratio:.12g}\n\n"
        )

    handle.write("* --- Neuron core ---\n")
    handle.write("XU1 node3 node1 vdd vee node2 TLV2372\n")
    handle.write("C1 node1 node2 1n\n")
    handle.write("R5 node3 0 10k\n")
    handle.write("R1 node2 node4 4.7k\n")
    handle.write("XU2 node4 0 vdd vee node6 TLV2372\n")
    handle.write("R4 node6 node7 1k\n")
    handle.write("R3 node7 node8 100\n")
    handle.write("D1 node8 node1 DI_1N4001G\n")
    handle.write("R2 node4 node7 9.4k\n")
    handle.write("XCMP node7 0 cmpout vddl vddc cmp\n")
    handle.write("X1 cmpout vout edge_to_pulse vth=0.5 vhigh=1.0 width=1u\n")
    handle.write("Xspkout vout vout_SRC spikegen vth=0.8 a=-0.20 b=0.0 Tr=3u\n")
    handle.write(".ends single_neuron\n\n")

    # Retained from the MATLAB generator even though it is not instantiated by
    # the current top-level circuit.
    handle.write(".subckt current_mirror vout_SRC TOP_SRC1 node1_inv\n")
    handle.write("M_1 vout_SRC vout_SRC 0 0 nmos_3p3 L=0.35u W=20u\n")
    handle.write("R_1 TOP_SRC1 TOP_SRC2 500k\n")
    handle.write("M_2 TOP_SRC2 vout_SRC node1_inv 0 nmos_3p3 L=0.35u W=20u\n")
    handle.write(".ends current_mirror\n\n")


def _write_spice_control(
    handle: TextIO,
    n_inputs: int,
    n_neurons: int,
    timestep_us: float,
    simulation_us: float,
    save_period_us: float,
    write_spice_print: bool,
) -> None:
    signals = expected_signals(n_inputs, n_neurons)
    handle.write("* === Selected Spectre Raw Outputs ===\n")
    handle.write("simulator lang=spectre\n")
    handle.write(
        "simOptions options save=selected currents=selected "
        "saveselectedtoallpub=nooutput\n"
    )
    for signal in signals:
        handle.write(f"save {signal}\n")
    handle.write("simulator lang=spice\n\n")

    _write_initial_conditions(handle, n_neurons)
    handle.write("* === Transient Analysis ===\n")
    handle.write("simulator lang=spectre\n")
    handle.write(
        f"tran_main tran stop={simulation_us:.12g}u "
        f"maxstep={timestep_us:.12g}u "
        f"strobeperiod={save_period_us:.12g}u "
        "strobeoutput=strobeonly\n"
    )
    handle.write("simulator lang=spice\n\n")
    if write_spice_print:
        handle.write("* === Optional SPICE Print Output ===\n")
        handle.write(".print tran \\\n")
        for index, signal in enumerate(signals):
            suffix = " \\\n" if index < len(signals) - 1 else "\n"
            handle.write(f"+ V({signal}){suffix}")
        handle.write("\n")
    handle.write("\n.end\n")


def generate_spice_model_inhabi(
    filename: str | Path = "stdp.sp",
    rootpath: str | Path | None = None,
    n_inputs: int = 3,
    n_neurons: int = 2,
    base_gap: float = 1.0e-9,
    gap_sigma: float = 0.05,
    timestep_us: float = DEFAULT_TIMESTEP_US,
    simulation_us: float = DEFAULT_SIMULATION_US,
    save_period_us: float | None = DEFAULT_SAVE_PERIOD_US,
    seed: int | None = None,
    write_spice_print: bool = False,
    input_pwl: str | Path | None = None,
    adaptive_a_ref: float = DEFAULT_ADAPTIVE_A_REF,
    adaptive_r_ref: float = DEFAULT_ADAPTIVE_R_REF,
    adaptive_a_min_mag: float = DEFAULT_ADAPTIVE_A_MIN_MAG,
    adaptive_a_max_mag: float = DEFAULT_ADAPTIVE_A_MAX_MAG,
    set_dg_equalization_exponent: float = DEFAULT_SET_DG_EQUALIZATION_EXPONENT,
    reset_dg_equalization_exponent: float = DEFAULT_RESET_DG_EQUALIZATION_EXPONENT,
    dg_equalization_curvature: float = DEFAULT_DG_EQUALIZATION_CURVATURE,
    learning_rate_scale: float = DEFAULT_LEARNING_RATE_SCALE,
    positive_dt_dg_ratio: float = DEFAULT_POSITIVE_DT_DG_RATIO,
    mem_threshold: float = DEFAULT_MEM_THRESHOLD,
) -> Path:
    """Generate the Spectre-compatible counterpart of the MATLAB SPICE deck."""
    if not RRAM_GAP_MIN_M <= base_gap <= RRAM_GAP_MAX_M:
        raise ValueError(
            f"base_gap must be between {RRAM_GAP_MIN_M:g} and "
            f"{RRAM_GAP_MAX_M:g} m"
        )
    if gap_sigma < 0.0:
        raise ValueError("gap_sigma must be zero or positive")
    if save_period_us is None:
        save_period_us = timestep_us
    if save_period_us <= 0.0:
        raise ValueError("save_period_us must be positive")
    save_intervals = simulation_us / save_period_us
    if not math.isclose(
        save_intervals,
        round(save_intervals),
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError(
            "simulation_us must be exactly divisible by save_period_us so "
            "the final RRAM state is saved"
        )
    if not 0.0 <= positive_dt_dg_ratio <= 1.0:
        raise ValueError("positive_dt_dg_ratio must be between 0 and 1")
    root = Path(rootpath) if rootpath is not None else Path(__file__).resolve().parent
    encoded_input = (
        Path(input_pwl) if input_pwl is not None else root / DEFAULT_INPUT_PWL
    )
    output = Path(filename)
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    with output.open("w", encoding="utf-8", newline="\n") as handle:
        _write_header(handle, root, encoded_input)
        _write_power(handle, mem_threshold)
        _write_neurons(handle, n_inputs, n_neurons, base_gap, gap_sigma, rng)
        _write_lateral_connections(handle, n_neurons)
        _write_single_neuron(
            handle,
            n_inputs,
            timestep_us,
            adaptive_a_ref,
            adaptive_r_ref,
            adaptive_a_min_mag,
            adaptive_a_max_mag,
            set_dg_equalization_exponent,
            reset_dg_equalization_exponent,
            dg_equalization_curvature,
            learning_rate_scale,
            positive_dt_dg_ratio,
        )
        _write_spice_control(
            handle,
            n_inputs,
            n_neurons,
            timestep_us,
            simulation_us,
            save_period_us,
            write_spice_print,
        )

    print(
        f"Generated {output} (N={n_inputs} inputs, M={n_neurons} neurons, "
        f"bounded Gaussian sigma={gap_sigma * 100:.1f}% over "
        f"{RRAM_GAP_MIN_M * 1e9:g}-{RRAM_GAP_MAX_M * 1e9:g} nm, "
        f"positive-delta-t delta-G ratio={positive_dt_dg_ratio:g}, "
        f"save period={save_period_us:g} us (strobe-only), "
        "Spectre compatible)"
    )
    return output


def run_spectre(
    netlist: Path,
    executable: str,
    workdir: Path,
    aps: bool = True,
    threads: int = DEFAULT_THREADS,
) -> Path:
    spectre = shutil.which(executable) if "/" not in executable else executable
    if not spectre:
        raise FileNotFoundError(
            f"Cannot find Spectre executable '{executable}'. Use --spectre /path/to/spectre."
        )

    raw_path = workdir / "out.raw"
    log_path = workdir / "spectre.log"
    # A previous SPICE-print run can leave a very large file behind.  The
    # selected Nutmeg raw result is the Python flow's source of truth.
    netlist.with_suffix(".print").unlink(missing_ok=True)

    command = [str(spectre)]
    if aps:
        command.append("++aps")
    if threads > 0:
        command.append(f"+mt={threads}")
    command.extend([
        str(netlist.resolve()),
        "-format",
        "nutascii",
        "-raw",
        str(raw_path.resolve()),
        "+log",
        str(log_path.resolve()),
    ])
    print("Running:", " ".join(command))
    completed = subprocess.run(command, cwd=workdir, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Spectre exited with status {completed.returncode}. See {log_path} for details."
        )
    return raw_path


def infer_raw_dimensions(raw_path: str | Path) -> tuple[int, int]:
    """Infer input and neuron counts from a Spectre Nutmeg raw header."""
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Spectre raw file not found: {path}")

    neuron_ids: set[int] = set()
    input_ids: set[int] = set()
    pattern = re.compile(r"\bXNEU(?P<neuron>\d+)\.R_out(?P<input>\d+)\b", re.I)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("Values:"):
                break
            match = pattern.search(line)
            if match:
                neuron_ids.add(int(match.group("neuron")))
                input_ids.add(int(match.group("input")))

    if not neuron_ids or not input_ids:
        raise ValueError(f"Cannot infer XNEU#.R_out# dimensions from {path}")
    expected_neurons = set(range(1, max(neuron_ids) + 1))
    expected_inputs = set(range(1, max(input_ids) + 1))
    if neuron_ids != expected_neurons or input_ids != expected_inputs:
        raise ValueError(f"Non-contiguous neuron or RRAM signal IDs in {path}")
    return len(input_ids), len(neuron_ids)


def plot_results(
    data: "np.ndarray",
    n_inputs: int,
    n_neurons: int,
    figure_dir: str | Path,
    show: bool = True,
    save_static: bool = True,
    animate_weights: bool = True,
    animation_frames: int = 80,
    animation_fps: int = 10,
) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FuncAnimation, PillowWriter

    output_dir = Path(figure_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_figures: list[Path] = []

    time = data[:, 0]
    neuron_stride = 2 + n_inputs
    side = int(round(n_inputs**0.5))
    if side * side != n_inputs:
        raise ValueError("Weight-map plotting requires a square number of inputs")

    frame_indices = np.unique(
        np.searchsorted(
            time,
            np.linspace(time[0], time[-1], animation_frames),
            side="left",
        ).clip(max=time.size - 1)
    )
    # Use one fixed scale for every neuron and frame so color changes represent
    # real conductance changes rather than per-frame color normalization.
    sampled_conductance_us: list[np.ndarray] = []
    for neuron in range(1, n_neurons + 1):
        start = 1 + (neuron - 1) * neuron_stride
        sampled_resistance = data[
            frame_indices, start + 2 : start + 2 + n_inputs
        ]
        sampled_conductance_us.append(1.0e6 / sampled_resistance)
    color_min = min(float(np.nanmin(values)) for values in sampled_conductance_us)
    color_max = max(float(np.nanmax(values)) for values in sampled_conductance_us)
    if not np.isfinite(color_min) or not np.isfinite(color_max):
        raise ValueError("Non-finite RRAM conductance encountered while plotting")
    if color_max <= color_min:
        color_max = color_min + max(abs(color_min) * 1.0e-6, 1.0e-9)

    for neuron in range(1, n_neurons + 1):
        start = 1 + (neuron - 1) * neuron_stride
        vout = data[:, start]
        vmem = data[:, start + 1]
        resistances = data[:, start + 2 : start + 2 + n_inputs]

        if save_static:
            figure, axes = plt.subplots(3, 1, num=neuron, figsize=(8, 10))
            weights_us = np.reshape(
                1.0e6 / resistances[-1, :], (side, side), order="F"
            )
            image = axes[0].imshow(
                weights_us,
                cmap="viridis",
                aspect="equal",
                vmin=color_min,
                vmax=color_max,
            )
            figure.colorbar(image, ax=axes[0], label="Conductance (µS)")
            axes[0].set_title(f"Neuron {neuron} - RRAM Conductance Map")

            axes[1].plot(time, 1.0e6 / resistances)
            axes[1].set_title(f"Neuron {neuron} - RRAM Conductance Evolution")
            axes[1].set_xlabel("Time (s)")
            axes[1].set_ylabel("Conductance (µS)")

            axes[2].plot(time, np.column_stack((vout, vmem)))
            axes[2].set_title(f"Neuron {neuron} - Membrane Voltage")
            axes[2].set_xlabel("Time (s)")
            axes[2].set_ylabel("Voltage (V)")
            axes[2].legend(("vout_SRC", "node2"))
            figure.tight_layout()

            output_path = output_dir / f"neuron_{neuron}.png"
            figure.savefig(output_path, dpi=200, bbox_inches="tight")
            saved_figures.append(output_path)
            print(f"Saved {output_path}")

        if animate_weights:
            conductance_us = sampled_conductance_us[neuron - 1]
            frame_maps = np.stack(
                [
                    np.reshape(row, (side, side), order="F")
                    for row in conductance_us
                ]
            )
            animation_figure, axis = plt.subplots(figsize=(6.4, 5.8))
            animation_image = axis.imshow(
                frame_maps[0],
                cmap="viridis",
                aspect="equal",
                vmin=color_min,
                vmax=color_max,
            )
            animation_figure.colorbar(
                animation_image,
                ax=axis,
                label="Conductance (µS)",
            )
            axis.set_xlabel("Pixel column")
            axis.set_ylabel("Pixel row")
            axis.set_xticks(range(side))
            axis.set_yticks(range(side))
            title = axis.set_title("")

            def update(frame: int) -> tuple[object, object]:
                animation_image.set_data(frame_maps[frame])
                simulation_ms = time[frame_indices[frame]] * 1.0e3
                title.set_text(
                    f"Neuron {neuron} - RRAM Conductance Map\n"
                    f"Simulation time: {simulation_ms:.3f} ms"
                )
                return animation_image, title

            animation = FuncAnimation(
                animation_figure,
                update,
                frames=len(frame_indices),
                interval=1000.0 / animation_fps,
                blit=False,
            )
            animation_path = output_dir / f"neuron_{neuron}_weights.gif"
            animation.save(
                animation_path,
                writer=PillowWriter(fps=animation_fps),
                dpi=110,
            )
            plt.close(animation_figure)
            saved_figures.append(animation_path)
            print(f"Saved {animation_path}")

    if show:
        plt.show()
    else:
        plt.close("all")
    return saved_figures


def _build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectre", default="spectre", help="Spectre executable or absolute path")
    parser.add_argument("--workdir", type=Path, default=root / "temp")
    parser.add_argument("--inputs", type=int, default=DEFAULT_N)
    parser.add_argument("--neurons", type=int, default=DEFAULT_M)
    parser.add_argument("--timestep-us", type=float, default=DEFAULT_TIMESTEP_US)
    parser.add_argument(
        "--save-period-us",
        type=float,
        default=DEFAULT_SAVE_PERIOD_US,
        help=(
            "uniform interval for saved Spectre samples; internal solver "
            "steps are retained (default: --timestep-us)"
        ),
    )
    parser.add_argument(
        "--simulation-us",
        type=float,
        help=(
            "transient duration in us; by default use the total duration recorded "
            f"in {DEFAULT_INPUT_PWL} (fallback: {DEFAULT_SIMULATION_US:g} us)"
        ),
    )
    parser.add_argument(
        "--input-pwl",
        type=Path,
        default=root / DEFAULT_INPUT_PWL,
        help=f"PWL dataset generated by main_encoding.py (default: {DEFAULT_INPUT_PWL})",
    )
    parser.add_argument("--base-gap", type=float, default=1.0e-9)
    parser.add_argument("--gap-sigma", type=float, default=0.01)
    parser.add_argument(
        "--mem-threshold",
        type=float,
        default=DEFAULT_MEM_THRESHOLD,
        help=(
            "comparator low rail controlling easy reset/firing "
            f"(default: {DEFAULT_MEM_THRESHOLD:.2f} V; original: 0.53 V)"
        ),
    )
    parser.add_argument(
        "--adaptive-a-ref",
        type=float,
        default=DEFAULT_ADAPTIVE_A_REF,
        help=(
            "synaptic ramp a at the reference resistance "
            f"(default: {DEFAULT_ADAPTIVE_A_REF:g} V)"
        ),
    )
    parser.add_argument(
        "--adaptive-r-ref",
        type=float,
        default=DEFAULT_ADAPTIVE_R_REF,
        help="RRAM resistance corresponding to adaptive-a-ref, in ohms",
    )
    parser.add_argument(
        "--adaptive-a-min-mag",
        type=float,
        default=DEFAULT_ADAPTIVE_A_MIN_MAG,
        help="minimum magnitude of the resistance-adaptive ramp start voltage",
    )
    parser.add_argument(
        "--adaptive-a-max-mag",
        type=float,
        default=DEFAULT_ADAPTIVE_A_MAX_MAG,
        help="maximum magnitude of the resistance-adaptive ramp start voltage",
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
    parser.add_argument(
        "--learning-rate-scale",
        type=float,
        default=DEFAULT_LEARNING_RATE_SCALE,
        help=(
            "base RRAM state-update multiplier before branch asymmetry "
            f"(default: {DEFAULT_LEARNING_RATE_SCALE:g})"
        ),
    )
    parser.add_argument(
        "--positive-dt-dg-ratio",
        type=float,
        default=DEFAULT_POSITIVE_DT_DG_RATIO,
        help=(
            "RESET/update-rate ratio for the t_post - t_pre > 0 "
            "conductance-decrease branch; 0 disables RESET and 1 restores "
            f"symmetric learning (default: {DEFAULT_POSITIVE_DT_DG_RATIO:g})"
        ),
    )
    parser.add_argument("--seed", type=int, help="optional reproducible Gaussian seed")
    parser.add_argument(
        "--aps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable Spectre ++aps acceleration (default: enabled)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help=f"Spectre worker threads (default: {DEFAULT_THREADS}; 0 disables +mt)",
    )
    parser.add_argument(
        "--spice-print",
        action="store_true",
        help="also write stdp.print; disabled by default because it can be very large",
    )
    parser.add_argument("--no-run", action="store_true", help="generate stdp.sp without running Spectre")
    parser.add_argument(
        "--figure-dir",
        type=Path,
        help="PNG/GIF output directory (default: WORKDIR/figures)",
    )
    parser.add_argument(
        "--plot-existing",
        action="store_true",
        help="plot WORKDIR/out.raw without regenerating or running the netlist",
    )
    parser.add_argument(
        "--animate-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save a time-resolved RRAM conductance-map GIF per neuron",
    )
    parser.add_argument(
        "--animation-only",
        action="store_true",
        help="save weight-map GIFs without regenerating the static PNG figures",
    )
    parser.add_argument(
        "--animation-frames",
        type=int,
        default=80,
        help="number of uniformly spaced simulation times per GIF (default: 80)",
    )
    parser.add_argument(
        "--animation-fps",
        type=int,
        default=10,
        help="animated GIF playback rate (default: 10 frames/s)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="save figures without opening interactive windows",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="run and parse without saving or opening figures",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.threads < 0:
        raise ValueError("--threads must be zero or positive")
    if args.timestep_us <= 0.0:
        raise ValueError("--timestep-us must be positive")
    if args.save_period_us is not None and args.save_period_us <= 0.0:
        raise ValueError("--save-period-us must be positive")
    if not RRAM_GAP_MIN_M <= args.base_gap <= RRAM_GAP_MAX_M:
        raise ValueError("--base-gap must be between 0.9e-9 and 1.1e-9 m")
    if args.gap_sigma < 0.0:
        raise ValueError("--gap-sigma must be zero or positive")
    if args.animation_frames < 2:
        raise ValueError("--animation-frames must be at least 2")
    if args.animation_fps <= 0:
        raise ValueError("--animation-fps must be positive")
    if args.animation_only and not args.animate_weights:
        raise ValueError("--animation-only requires --animate-weights")
    if args.adaptive_r_ref <= 0.0:
        raise ValueError("--adaptive-r-ref must be positive")
    if args.adaptive_a_min_mag < 0.0:
        raise ValueError("--adaptive-a-min-mag must be zero or positive")
    if args.adaptive_a_max_mag <= 0.0:
        raise ValueError("--adaptive-a-max-mag must be positive")
    if args.adaptive_a_min_mag > args.adaptive_a_max_mag:
        raise ValueError("--adaptive-a-min-mag cannot exceed --adaptive-a-max-mag")
    if args.set_dg_equalization_exponent <= 0.0:
        raise ValueError("--set-dg-equalization-exponent must be positive")
    if args.reset_dg_equalization_exponent <= 0.0:
        raise ValueError("--reset-dg-equalization-exponent must be positive")
    if args.dg_equalization_curvature < 0.0:
        raise ValueError("--dg-equalization-curvature cannot be negative")
    if args.learning_rate_scale < 0.0:
        raise ValueError("--learning-rate-scale must be zero or positive")
    if not 0.0 <= args.positive_dt_dg_ratio <= 1.0:
        raise ValueError("--positive-dt-dg-ratio must be between 0 and 1")
    if not 0.0 <= args.mem_threshold < 0.5:
        raise ValueError(
            "--mem-threshold must be at least 0 V and below the 0.5 V "
            "edge-detector threshold"
        )
    root = Path(__file__).resolve().parent
    workdir = args.workdir.expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    plot_inputs = args.inputs
    plot_neurons = args.neurons
    if args.plot_existing:
        if args.no_run:
            raise ValueError("--plot-existing cannot be combined with --no-run")
        raw_path = workdir / "out.raw"
        plot_inputs, plot_neurons = infer_raw_dimensions(raw_path)
        print(
            f"Plotting existing {raw_path} "
            f"({plot_inputs} inputs, {plot_neurons} neurons)"
        )
    else:
        input_pwl = args.input_pwl.expanduser().resolve()
        source_count, dataset_duration_us = inspect_input_pwl(input_pwl, args.inputs)
        simulation_us = args.simulation_us
        if simulation_us is None:
            simulation_us = dataset_duration_us or DEFAULT_SIMULATION_US
        dataset_duration_text = (
            f", encoded duration={dataset_duration_us:g} us"
            if dataset_duration_us is not None
            else ""
        )
        print(
            f"Using encoded input {input_pwl} "
            f"({source_count} PWL sources{dataset_duration_text}); "
            f"simulation={simulation_us:g} us"
        )
        netlist = generate_spice_model_inhabi(
            filename=workdir / "stdp.sp",
            rootpath=root,
            n_inputs=args.inputs,
            n_neurons=args.neurons,
            base_gap=args.base_gap,
            gap_sigma=args.gap_sigma,
            timestep_us=args.timestep_us,
            simulation_us=simulation_us,
            save_period_us=args.save_period_us,
            seed=args.seed,
            write_spice_print=args.spice_print,
            input_pwl=input_pwl,
            adaptive_a_ref=args.adaptive_a_ref,
            adaptive_r_ref=args.adaptive_r_ref,
            adaptive_a_min_mag=args.adaptive_a_min_mag,
            adaptive_a_max_mag=args.adaptive_a_max_mag,
            set_dg_equalization_exponent=args.set_dg_equalization_exponent,
            reset_dg_equalization_exponent=args.reset_dg_equalization_exponent,
            dg_equalization_curvature=args.dg_equalization_curvature,
            learning_rate_scale=args.learning_rate_scale,
            positive_dt_dg_ratio=args.positive_dt_dg_ratio,
            mem_threshold=args.mem_threshold,
        )
        if args.no_run:
            return
        raw_path = run_spectre(netlist, args.spectre, workdir, args.aps, args.threads)

    from import_data import import_data

    data, _ = import_data(raw_path, expected_signals(plot_inputs, plot_neurons))
    print(f"Read {data.shape[0]} transient points from {raw_path}")
    if not args.no_plot:
        figure_dir = args.figure_dir or workdir / "figures"
        plot_results(
            data,
            plot_inputs,
            plot_neurons,
            figure_dir,
            show=not args.no_show,
            save_static=not args.animation_only,
            animate_weights=args.animate_weights,
            animation_frames=args.animation_frames,
            animation_fps=args.animation_fps,
        )


if __name__ == "__main__":
    main()
