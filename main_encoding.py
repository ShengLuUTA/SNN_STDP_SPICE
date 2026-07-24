#!/usr/bin/env python3
"""MNIST -> 8x8 -> Poisson spike trains -> Spectre-compatible PWL sources.

This is the Python 3 counterpart of ``main_encoding.m``.  It retains the
original data shapes, balanced digit sampling, shuffle constraint, voltage
levels, and edge timing.  Redundant collinear PWL points are omitted by
default so Spectre is not forced to stop every 0.1 us while the input is flat.
"""

from __future__ import annotations

import argparse
import gzip
import math
import os
import shutil
import struct
import tempfile
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
from PIL import Image


IMAGE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz"
LABEL_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz"


def _download_and_unpack(url: str, gzip_path: Path, raw_path: Path) -> None:
    if raw_path.is_file():
        print(f"{raw_path.name} exists. Skipping download.")
        return
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, gzip_path)
    with gzip.open(gzip_path, "rb") as source, raw_path.open("wb") as target:
        shutil.copyfileobj(source, target)


def load_mnist_images(path: str | Path) -> np.ndarray:
    """Load an IDX image file as ``[28, 28, image]`` uint8 data."""
    path = Path(path)
    with path.open("rb") as handle:
        header = handle.read(16)
        if len(header) != 16:
            raise ValueError(f"Incomplete MNIST image header: {path}")
        magic, count, rows, columns = struct.unpack(">IIII", header)
        if magic != 2051:
            raise ValueError(f"Unexpected MNIST image magic {magic} in {path}")
        raw = handle.read()
    images = np.frombuffer(raw, dtype=np.uint8)
    expected = count * rows * columns
    if images.size != expected:
        raise ValueError(f"Expected {expected} image bytes in {path}, found {images.size}")
    return images.reshape(count, rows, columns).transpose(1, 2, 0)


def load_mnist_labels(path: str | Path) -> np.ndarray:
    """Load an IDX label file as a one-dimensional uint8 array."""
    path = Path(path)
    with path.open("rb") as handle:
        header = handle.read(8)
        if len(header) != 8:
            raise ValueError(f"Incomplete MNIST label header: {path}")
        magic, count = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError(f"Unexpected MNIST label magic {magic} in {path}")
        labels = np.frombuffer(handle.read(), dtype=np.uint8)
    if labels.size != count:
        raise ValueError(f"Expected {count} labels in {path}, found {labels.size}")
    return labels.copy()


def _matlab_disk_order(array: np.ndarray) -> np.ndarray:
    if array.ndim == 1:
        # MATLAB arrays have at least two dimensions. A MATLAB column vector
        # [N, 1] is stored by v7.3/HDF5 with reversed dimensions [1, N].
        return array[np.newaxis, :]
    if array.ndim < 2:
        return array
    return array.transpose(tuple(reversed(range(array.ndim))))


def load_mat_variables(path: str | Path, *names: str) -> tuple[np.ndarray, ...]:
    """Load MATLAB 7.3/HDF5 arrays and restore MATLAB dimension order."""
    arrays: list[np.ndarray] = []
    with h5py.File(path, "r") as handle:
        for name in names:
            if name not in handle:
                raise KeyError(f"Variable '{name}' is absent from {path}")
            value = np.asarray(handle[name])
            if value.ndim >= 2:
                value = value.transpose(tuple(reversed(range(value.ndim))))
            arrays.append(np.squeeze(value) if value.ndim == 2 and 1 in value.shape else value)
    return tuple(arrays)


def save_mat_v73(path: str | Path, variables: dict[str, tuple[np.ndarray, str]]) -> None:
    """Write the small MATLAB 7.3 subset used by this project."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with h5py.File(temporary, "w", userblock_size=512) as handle:
            for name, (array, matlab_class) in variables.items():
                disk_array = _matlab_disk_order(np.asarray(array))
                dataset = handle.create_dataset(
                    name,
                    data=disk_array,
                    compression="gzip" if disk_array.size > 1024 else None,
                    shuffle=disk_array.size > 1024,
                )
                dataset.attrs["MATLAB_class"] = np.bytes_(matlab_class)
                if matlab_class == "logical":
                    dataset.attrs["MATLAB_int_decode"] = np.int32(1)

        _write_matlab_header(temporary)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_matlab_header(path: Path) -> None:
    """Write the 128-byte MATLAB 7.3 user-block header."""

    description = (
        "MATLAB 7.3 MAT-file, Platform: Python, Created on: "
        f"{datetime.now():%a %b %d %H:%M:%S %Y} HDF5 schema 1.00 ."
    ).encode("ascii")[:116]
    header = description.ljust(116, b" ") + (b"\x00" * 8) + b"\x00\x02IM"
    with path.open("r+b") as handle:
        handle.write(header)


def save_poisson_spikes_mat_v73(
    path: str | Path,
    images8: np.ndarray,
    labels: np.ndarray,
    time_steps: int,
    max_spikes: float,
    rng: np.random.Generator,
    batch_size: int = 256,
) -> None:
    """Generate and stream all spike trains without holding the full tensor in RAM."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_count = images8.shape[2]
    pixels = images8.reshape(64, image_count, order="F")
    probabilities = pixels * max_spikes / time_steps

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with h5py.File(temporary, "w", userblock_size=512) as handle:
            # MATLAB [64, T, N] arrays appear as [N, T, 64] through h5py.
            spike_chunks = (
                min(batch_size, image_count),
                min(time_steps, 256),
                64,
            )
            spike_dataset = handle.create_dataset(
                "spikes",
                shape=(image_count, time_steps, 64),
                dtype=np.uint8,
                chunks=spike_chunks,
                compression="gzip",
                shuffle=True,
            )
            spike_dataset.attrs["MATLAB_class"] = np.bytes_("logical")
            spike_dataset.attrs["MATLAB_int_decode"] = np.int32(1)

            images_dataset = handle.create_dataset(
                "images8",
                data=_matlab_disk_order(images8),
                compression="gzip",
                shuffle=True,
            )
            images_dataset.attrs["MATLAB_class"] = np.bytes_("double")

            labels_dataset = handle.create_dataset(
                "labels_spike",
                data=_matlab_disk_order(np.asarray(labels, dtype=float)),
                compression="gzip",
                shuffle=True,
            )
            labels_dataset.attrs["MATLAB_class"] = np.bytes_("double")

            for start in range(0, image_count, batch_size):
                stop = min(start + batch_size, image_count)
                draws = rng.random((64, time_steps, stop - start))
                spike_block = draws < probabilities[:, None, start:stop]
                spike_dataset[start:stop, :, :] = spike_block.transpose(2, 1, 0)

        _write_matlab_header(temporary)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_spike_samples(path: str | Path, indices: Sequence[int]) -> np.ndarray:
    """Load selected zero-based samples as MATLAB-order ``[64, T, N]`` data."""
    requested = np.asarray(indices, dtype=np.int64)
    with h5py.File(path, "r") as handle:
        if "spikes" not in handle:
            raise KeyError(f"Variable 'spikes' is absent from {path}")
        dataset = handle["spikes"]
        if dataset.ndim != 3 or dataset.shape[2] != 64:
            raise ValueError(f"Unexpected spike dataset shape {dataset.shape} in {path}")
        if np.any(requested < 0) or np.any(requested >= dataset.shape[0]):
            raise IndexError(f"Spike sample index is outside [0, {dataset.shape[0] - 1}]")
        selected = np.empty((64, dataset.shape[1], requested.size), dtype=bool)
        for output_index, source_index in enumerate(requested):
            selected[:, :, output_index] = np.asarray(
                dataset[int(source_index), :, :], dtype=bool
            ).T
    return selected


def resize_images(images: np.ndarray) -> np.ndarray:
    """Resize ``[28, 28, N]`` images to normalized ``[8, 8, N]`` data."""
    count = images.shape[2]
    output = np.empty((8, 8, count), dtype=np.float64)
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    for index in range(count):
        image = Image.fromarray(images[:, :, index], mode="L")
        output[:, :, index] = np.asarray(image.resize((8, 8), resampling), dtype=float) / 255.0
    return output


def poisson_encode(
    images8: np.ndarray,
    time_steps: int,
    max_spikes: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return logical spikes with MATLAB-compatible ``[64, T, N]`` shape."""
    image_count = images8.shape[2]
    pixels = images8.reshape(64, image_count, order="F")
    probabilities = pixels * max_spikes / time_steps
    spikes = np.empty((64, time_steps, image_count), dtype=bool)
    for start in range(0, image_count, 256):
        stop = min(start + 256, image_count)
        draws = rng.random((64, time_steps, stop - start))
        spikes[:, :, start:stop] = draws < probabilities[:, None, start:stop]
    return spikes


def generate_sp_file(
    idx: int,
    matfile: str | Path,
    outfile: str | Path,
    v_high: float,
    v_low: float,
    points_per_us: int = 10,
    compact: bool = True,
    dt_us: float = 1.0,
) -> None:
    """Generate PWL sources for one sample; ``idx`` remains MATLAB-style 1-based."""
    (labels,) = load_mat_variables(matfile, "labels_spike")
    if idx < 1 or idx > labels.size:
        raise IndexError(f"Sample index must be in [1, {labels.size}]")
    spike_sample = load_spike_samples(matfile, [idx - 1])[:, :, 0]
    label = int(labels[idx - 1])
    _write_pwl_sources(
        outfile,
        spike_sample[:, :, None],
        [label],
        v_high,
        v_low,
        points_per_us,
        [f"Spectre-compatible PWL file for MNIST sample {idx} (label {label})"],
        compact,
        dt_us,
    )


def _shuffle_without_adjacent_labels(
    indices: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    shuffled_indices = indices.copy()
    shuffled_labels = labels.copy()
    for _ in range(100):
        order = rng.permutation(indices.size)
        shuffled_indices = indices[order]
        shuffled_labels = labels[order]
        if np.all(shuffled_labels[:-1] != shuffled_labels[1:]):
            break
    return shuffled_indices, shuffled_labels


def _write_pwl_sources(
    outfile: str | Path,
    selected_spikes: np.ndarray,
    selected_labels: Sequence[int],
    v_high: float,
    v_low: float,
    points_per_us: int,
    header_lines: Sequence[str],
    compact: bool = True,
    dt_us: float = 1.0,
) -> None:
    if points_per_us <= 0:
        raise ValueError("points_per_us must be positive")
    if dt_us <= 0.0:
        raise ValueError("dt_us must be positive")
    pixels, time_steps, sample_count = selected_spikes.shape
    edge_step_us = 1.0 / points_per_us
    if dt_us + 1.0e-12 < edge_step_us:
        raise ValueError(
            f"dt_us={dt_us:g} is shorter than the {edge_step_us:g} us PWL edge; "
            "increase --points-per-us"
        )
    points_per_bin_exact = dt_us * points_per_us
    points_per_bin = int(round(points_per_bin_exact))
    if not compact and not math.isclose(
        points_per_bin_exact, points_per_bin, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise ValueError(
            "dense PWL output requires dt_us * points_per_us to be an integer"
        )
    total_duration_us = time_steps * sample_count * dt_us
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("w", encoding="utf-8", newline="\n") as handle:
        for line in header_lines:
            handle.write(f"* {line}\n")
        handle.write(f"* Total samples = {sample_count}\n")
        handle.write(f"* Label order = {list(map(int, selected_labels))}\n")
        handle.write(
            f"* Each spike point = {dt_us:g}us with "
            f"{edge_step_us:g}us edge resolution\n"
        )
        if compact:
            handle.write("* Redundant constant-voltage PWL points omitted losslessly\n\n")
        else:
            handle.write(f"* Dense output: {points_per_us} points per us\n\n")

        for pixel in range(pixels):
            handle.write(f"Vin{pixel + 1} vin{pixel + 1} 0 PWL(\n")
            step_values = np.where(
                selected_spikes[pixel].T.reshape(sample_count * time_steps),
                v_high,
                v_low,
            )
            if compact:
                transitions = np.flatnonzero(step_values[1:] != step_values[:-1]) + 1
                emitted: list[tuple[float, float]] = []

                def emit(time_us: float, voltage: float) -> None:
                    point = (float(time_us), float(voltage))
                    if not emitted or point != emitted[-1]:
                        handle.write(f"+ {time_us:.10g}u {voltage:.12g}\n")
                        emitted.append(point)

                emit(0.0, step_values[0])
                for transition in transitions:
                    transition_us = transition * dt_us
                    emit(
                        transition_us - edge_step_us,
                        step_values[transition - 1],
                    )
                    emit(transition_us, step_values[transition])
                emit(total_duration_us - edge_step_us, step_values[-1])
            else:
                for index, voltage in enumerate(step_values):
                    base_time_us = index * dt_us
                    for point in range(points_per_bin):
                        time_us = base_time_us + edge_step_us * point
                        handle.write(f"+ {time_us:.10g}u {voltage:.12g}\n")
            handle.write(")\n\n")


def compact_existing_pwl(path: str | Path) -> tuple[int, int]:
    """Losslessly remove redundant flat points from an existing generated PWL deck."""
    source = Path(path)
    temporary = source.with_name(f".{source.name}.compact.tmp")
    original_points = 0
    compact_points = 0

    try:
        with source.open("r", encoding="utf-8") as reader, temporary.open(
            "w", encoding="utf-8", newline="\n"
        ) as writer:
            marker = "* Losslessly compacted: redundant constant-voltage PWL points removed\n"
            writer.write(marker)
            iterator = iter(reader)
            for line in iterator:
                if line == marker:
                    continue
                if line.startswith("* Each spike point =") and "expanded" in line:
                    line = "* Each spike point = 1us with 0.1us edge resolution\n"
                writer.write(line)
                if " PWL(" not in line.upper():
                    continue

                points: list[str] = []
                values: list[str] = []
                for point_line in iterator:
                    if point_line.strip() == ")":
                        break
                    if point_line.lstrip().startswith("+"):
                        fields = point_line.split()
                        if len(fields) < 3:
                            raise ValueError(f"Malformed PWL point in {source}: {point_line!r}")
                        points.append(point_line)
                        values.append(fields[2].casefold())
                    else:
                        writer.write(point_line)

                if not points:
                    raise ValueError(f"Empty PWL source in {source}")
                keep = {0, len(points) - 1}
                for index in range(1, len(points)):
                    if values[index] != values[index - 1]:
                        keep.update((index - 1, index))
                for index in sorted(keep):
                    writer.write(points[index])
                writer.write(")\n")
                original_points += len(points)
                compact_points += len(keep)
        temporary.replace(source)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return original_points, compact_points


def generate_sp_file_samples(
    matfile: str | Path,
    outfile: str | Path,
    v_high: float,
    v_low: float,
    target: Sequence[int],
    numsample: int,
    rng: np.random.Generator | None = None,
    points_per_us: int = 10,
    compact: bool = True,
    dt_us: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate balanced, shuffled multi-sample PWL voltage sources."""
    if rng is None:
        rng = np.random.default_rng()
    (labels,) = load_mat_variables(matfile, "labels_spike")
    labels = labels.astype(np.uint8, copy=False)

    index_groups: list[np.ndarray] = []
    label_groups: list[np.ndarray] = []
    for digit in target:
        matching = np.flatnonzero(labels == digit)
        if matching.size < numsample:
            raise ValueError(f"Not enough samples for class {digit}")
        chosen = matching[:numsample]
        index_groups.append(chosen)
        label_groups.append(labels[chosen])

    indices = np.concatenate(index_groups)
    selected_labels = np.concatenate(label_groups)
    indices, selected_labels = _shuffle_without_adjacent_labels(indices, selected_labels, rng)
    selected_spikes = load_spike_samples(matfile, indices)

    sample_us = selected_spikes.shape[1] * dt_us
    _write_pwl_sources(
        outfile,
        selected_spikes,
        selected_labels,
        v_high,
        v_low,
        points_per_us,
        [
            f"Spectre-compatible PWL file for MNIST digits: {list(target)}",
            f"Balanced sampling: {numsample} samples per class",
            "Order shuffled with no consecutive identical labels when found within 100 attempts",
            f"Each sample = {sample_us}us, total = {sample_us * indices.size}us",
        ],
        compact,
        dt_us,
    )
    print(f"Wrote {outfile}")
    return indices, selected_labels


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-us", type=float, default=100.0)
    parser.add_argument(
        "--dt-us",
        type=float,
        default=1.0,
        help="spike-bin duration in us; fractional values such as 0.5 are supported",
    )
    parser.add_argument("--max-spikes", type=float, default=40.0)
    parser.add_argument("--targets", type=int, nargs="+", default=[0, 4, 7])
    parser.add_argument("--samples-per-class", type=int, default=20)
    parser.add_argument("--v-high", type=float, default=1.2)
    parser.add_argument("--v-low", type=float, default=0.0)
    parser.add_argument("--points-per-us", type=int, default=10)
    parser.add_argument(
        "--output-pwl",
        type=Path,
        help="output PWL deck (default: Full/mnist_digit1_20samples_7.sp)",
    )
    parser.add_argument(
        "--dense-pwl",
        action="store_true",
        help="write every PWL point instead of the default lossless compact form",
    )
    parser.add_argument(
        "--compact-existing-pwl",
        action="store_true",
        help="compact the existing generated PWL file in place without resampling data",
    )
    parser.add_argument("--seed", type=int, help="optional reproducible random seed")
    parser.add_argument(
        "--rebuild-spikes",
        action="store_true",
        help="regenerate mnist_spike_trains.mat using duration/dt/max-spikes/seed",
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--skip-pwl", action="store_true", help="prepare data without rewriting the PWL file")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    root = Path(__file__).resolve().parent
    pwl_file = (
        args.output_pwl.expanduser().resolve()
        if args.output_pwl is not None
        else root / "mnist_digit1_20samples_7.sp"
    )
    if args.compact_existing_pwl:
        before, after = compact_existing_pwl(pwl_file)
        print(f"Compacted {pwl_file}: {before} -> {after} PWL points")
        return

    if args.duration_us <= 0.0:
        raise ValueError("duration-us must be positive")
    if args.dt_us <= 0.0:
        raise ValueError("dt-us must be positive")
    time_steps_exact = args.duration_us / args.dt_us
    time_steps = int(round(time_steps_exact))
    if not math.isclose(time_steps_exact, time_steps, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("duration-us must be exactly divisible by dt-us")
    rng = np.random.default_rng(args.seed)

    image_gzip = root / "train-images-idx3-ubyte.gz"
    label_gzip = root / "train-labels-idx1-ubyte.gz"
    image_file = root / "train-images-idx3-ubyte"
    label_file = root / "train-labels-idx1-ubyte"
    print("Checking MNIST raw files...")
    _download_and_unpack(IMAGE_URL, image_gzip, image_file)
    _download_and_unpack(LABEL_URL, label_gzip, label_file)

    print("Loading MNIST images and labels...")
    images = load_mnist_images(image_file)
    labels = load_mnist_labels(label_file)
    if labels.size != images.shape[2]:
        raise ValueError("Label count does not match image count")

    dataset_file = root / "mnist_8x8_dataset.mat"
    if dataset_file.is_file():
        print("8x8 dataset already exists. Loading...")
        images8, labels8 = load_mat_variables(dataset_file, "images8", "labels8")
    else:
        print("Resizing to 8x8 and saving dataset...")
        images8 = resize_images(images)
        labels8 = labels.astype(float)
        save_mat_v73(
            dataset_file,
            {"images8": (images8, "double"), "labels8": (labels8, "double")},
        )

    spike_file = root / "mnist_spike_trains.mat"
    if spike_file.is_file() and not args.rebuild_spikes:
        print("Spike-train file already exists. Loading...")
        (labels_spike,) = load_mat_variables(spike_file, "labels_spike")
        with h5py.File(spike_file, "r") as handle:
            if "spikes" not in handle or handle["spikes"].ndim != 3:
                raise ValueError(f"Invalid spike dataset in {spike_file}")
            cached_time_steps = handle["spikes"].shape[1]
        if cached_time_steps != time_steps:
            raise ValueError(
                f"Cached {spike_file.name} has {cached_time_steps} time bins, but "
                f"duration-us={args.duration_us:g} and dt-us={args.dt_us:g} require "
                f"{time_steps}. Use --rebuild-spikes."
            )
    else:
        print("Generating rate-based Poisson spike trains...")
        labels_spike = labels.astype(float)
        save_poisson_spikes_mat_v73(
            spike_file,
            images8,
            labels_spike,
            time_steps,
            args.max_spikes,
            rng,
        )

    if not args.no_plot:
        import matplotlib.pyplot as plt

        sample = min(99, labels_spike.size - 1)
        spike_sample = load_spike_samples(spike_file, [sample])[:, :, 0]
        figure, axes = plt.subplots(1, 2)
        axes[0].imshow(images8[:, :, 0], cmap="gray")
        axes[0].set_title("8x8 Image")
        axes[1].imshow(spike_sample, cmap="gray", aspect="auto")
        axes[1].set_xlabel("Time step")
        axes[1].set_ylabel("Neuron Index")
        axes[1].set_title("Rate-based Poisson Spike Train")
        figure.tight_layout()
        plt.show()

    if not args.skip_pwl:
        generate_sp_file_samples(
            spike_file,
            pwl_file,
            args.v_high,
            args.v_low,
            args.targets,
            args.samples_per_class,
            rng=rng,
            points_per_us=args.points_per_us,
            compact=not args.dense_pwl,
            dt_us=args.dt_us,
        )


if __name__ == "__main__":
    main()
