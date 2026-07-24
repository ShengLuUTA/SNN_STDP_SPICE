# Python 3 and Spectre flow

The three MATLAB scripts now have direct Python 3 counterparts:

- `main_encoding.py` corresponds to `main_encoding.m`.
- `main_stdp.py` corresponds to `main_stdp.m`.
- `import_data.py` corresponds to `import_data.m`.

The MATLAB files remain unchanged for reference. The circuit topology,
subcircuit names, device values, MNIST encoding, balanced sampling, lateral
connections, saved RRAM values, and plotting layout are retained.

Python 3.10 or newer is recommended.

## Setup

```bash
cd Full
python3 -m pip install -r requirements.txt
```

Cadence Spectre must be available on `PATH`, or supplied with `--spectre`.

## Encode MNIST and generate the input PWL deck

```bash
python3 main_encoding.py --seed 1 --no-plot
```

Use `--no-plot` on a headless machine. Existing `.mat` files are reused. The
generated `mnist_digit1_20samples_7.sp` is ordinary SPICE PWL syntax, which
Spectre reads through its SPICE compatibility mode. By default, the Python
encoder writes only transition and edge points; this is waveform-equivalent to
the original dense 0.1 us expansion but avoids forcing Spectre to visit every
redundant flat point. Use `--dense-pwl` only when a dense reference deck is
specifically needed. An existing dense deck can be compacted without selecting
new MNIST samples:

```bash
python3 main_encoding.py --compact-existing-pwl
```

`main_stdp.py` includes this generated file directly. It validates that the
requested `Vin1` through `VinN` sources exist and, unless `--simulation-us` is
specified, uses the total encoded duration from the PWL header. To generate a
custom file and use that exact file in STDP:

```bash
python3 main_encoding.py --output-pwl inputs/my_mnist.sp --seed 1 --no-plot
python3 main_stdp.py --input-pwl inputs/my_mnist.sp --no-show
```

Use `--rebuild-spikes` when changing `--duration-us`, `--dt-us`,
`--max-spikes`, or the Poisson seed; otherwise the cached
`mnist_spike_trains.mat` is reused.
`--dt-us` accepts fractional spike-bin durations such as `0.5`; the sample
duration must be exactly divisible by the selected bin duration. For example:

```bash
python3 main_encoding.py --duration-us 100 --dt-us 0.5 --max-spikes 25 --targets 7 --samples-per-class 50 --seed 1 --rebuild-spikes --no-plot
```

## Generate and run the STDP deck

```bash
python3 main_stdp.py --spectre /path/to/spectre
```

The runner enables Spectre `++aps` with eight threads by default. Change the
thread count with `--threads N`, or use `--no-aps --threads 0` for a baseline
non-APS run.

Each run saves one PNG per neuron under `temp/figures/`. To save the figures
without opening interactive windows (for example, over SSH), run:

```bash
python3 main_stdp.py --no-show
```

Use `--figure-dir PATH` to choose a different output directory. `--no-plot`
disables both saving and interactive display.

Each run also saves `neuron_#_weights.gif`, a time-resolved 8x8 RRAM
conductance map with a fixed µS color scale across all neurons and frames. To
create only the animations from an existing simulation without rerunning
Spectre or rewriting `stdp.sp`, use:

```bash
python3 main_stdp.py --plot-existing --animation-only --no-show
```

Use `--animation-frames N` and `--animation-fps N` to control temporal detail
and playback speed. `--no-animate-weights` disables GIF generation.

To inspect only the generated Spectre-compatible deck:

```bash
python3 main_stdp.py --no-run
```

This writes `temp/stdp.sp`. Following `spectre_test/single_neuron.sp`, the deck
uses Spectre's SPICE-compatible format: it starts with
`simulator lang=spice`, uses `.include`, `.LIB`, `.tran`, and `.end`, and
switches to `simulator lang=spectre` around native `ahdl_include` and selected
`save` statements before returning to SPICE mode. The reference node is written
explicitly as node `0` throughout the generated circuit.

Spectre receives `-format nutascii`, so only the selected signals are written
to portable `temp/out.raw` for `import_data.py`. A separate `.print tran` file
is disabled by default because it duplicates the raw output and can become very
large; use `--spice-print` when that reference-format file is explicitly
needed.

The RRAM resistance output is saved through its connected subcircuit net
(`XNEU#.R_out#`). This is the Spectre form of the HSPICE terminal expression
`XNEU#.Xrram#.R_out`; it measures the same Verilog-A output voltage.

## Constant-delta-G adaptive synaptic ramp

Each synaptic input uses `adaptive_spikegen.va` instead of a fixed input ramp.
The module infers the RRAM gap from the live `R_out` value and solves the
nonlinear state equation

```text
G sinh(kappa(gap) V) = G_ref sinh(kappa(gap_ref) V_ref)
```

to calculate a different programming voltage for every resistance. Lower
resistance receives a lower voltage. Separate SET/RESET finite-pulse
corrections (`1.61` and `2.02`) plus a `0.30` log-resistance curvature term
equalize final delta-G over 0.9-1.1 nm. The controller dynamically adjusts both
the passed high level and the falling-edge ramp start; the neuron output
`Xspkout` retains the fixed `spikegen` behavior.

Defaults are `a_ref=-0.422 V`, `R_ref=263244 ohm`, `a_min=0.01 V`, and
`a_max=0.50 V`. They can be changed with `--adaptive-a-ref`,
`--adaptive-r-ref`, `--adaptive-a-min-mag`, and `--adaptive-a-max-mag`.

The RRAM model range is 0.9-1.1 nm. `main_stdp.py` bounds every Gaussian
initial-gap sample to this interval and rejects a base gap outside it. This
keeps the training deck inside the range used to calibrate the equal-delta-G
controller.

The generated training deck passes `learning_rate_scale=0.20` to `rram.va` as
the base update rate. SET uses that rate, while RESET uses the additional
`--positive-dt-dg-ratio` multiplier. Its default `0.40` makes the absolute
conductance decrease for positive delta-t approximately 40% of the negative
delta-t increase. Neither parameter changes the electrical I-V curve.

## Compact Spectre result saving

`main_stdp.py` uses Spectre transient strobing with
`strobeoutput=strobeonly`. Spectre continues taking every internal timestep
needed for circuit accuracy, but only uniform strobe points are written to
`out.raw`. `--save-period-us` controls the saved-data interval and defaults to
`--timestep-us`. For example, keep the 0.2 us simulation step limit while
saving only once per microsecond:

```bash
python3 main_stdp.py --timestep-us 0.2 --save-period-us 1 --threads 8 --no-show
```

For a 1000 us simulation this changes the nominal output from about 5001
points at 0.2 us to about 1001 points at 1 us. A save period no larger than the
1 us output-spike width is recommended when spike waveforms must remain
visible. The selected-node list is unchanged.
The simulation duration must be divisible by the selected save period so the
final RRAM state is always present in the raw result.

## Plot the STDP timing and initial-gap depth map

`plot_stdp_window.py` independently sweeps both the initial RRAM gap and paired
pre/post timing through the same `adaptive_spikegen.va`, `spikegen.va`, and
`rram.va` models used by the training flow. It defines
`delta-t = t_post - t_pre` at the falling threshold crossings that start the
programming ramps, so positive delta-t means pre first.

```bash
python3 plot_stdp_window.py --delta-min-us -4 --delta-max-us 4 --delta-step-us 0.25 --gap-min-nm 0.9 --gap-max-nm 1.1 --gap-step-nm 0.025 --plot-metric conductance --learning-rate-scale 0.20 --no-show
```

Use `--positive-dt-dg-ratio RATIO` to reduce the physical RRAM RESET update
rate corresponding to `t_post - t_pre > 0`. The default ratio `0.40` targets a
positive-branch absolute delta-G around 40% of the negative branch. The
accepted range is 0 through 1; use 1 to restore the symmetric window.

The script writes `stdp_window/stdp_window.png`, the numerical values in
`stdp_window/stdp_window.csv`, the generated Spectre deck, raw data, and log.
Because this sweep uses only the initial and final RRAM states, its raw output
is also strobed to those two endpoints instead of recording internal steps.
The default PNG contains a `delta-G` color-depth map and one `delta-G` versus
gap-time curve per initial gap. Gap time is shown in ns, and the initial gap is
shown in nm. Use `--plot-metric resistance` to plot `delta-R` instead. The CSV
contains both resistance and conductance values for every combination.

The standalone sweep uses `--adaptive-a-ref -0.422`,
`--adaptive-r-ref 263244`, and `--adaptive-a-max-mag 0.50`, together with the
same dynamic delta-G controller used by `main_stdp.py`. The calibration is for
the fixed 0.9-1.1 nm range and the default pulse/ramp timing.

Supplying `--base-gap 1e-9` overrides the gap range and restores the original
single-gap line plot. Waveform parameters such as `--pulse-width-us`,
`--pre-ramp-us`, `--post-ramp-us`, `--adaptive-a-min-mag`, and
`--adaptive-a-max-mag` can also be changed without modifying the training
code. Use `--no-run` to inspect only the generated Spectre deck.

## Membrane firing threshold control

The comparator low rail is configurable with `--mem-threshold`. It defaults
to `0.40 V`, reduced from the original `VDDC=0.53 V`. This puts the comparator
low state below the `edge_to_pulse` detector threshold of `0.50 V`, allowing a
clean reset and making subsequent rising-edge spike generation easier. For
example, use `--mem-threshold 0.35` for a more aggressive setting. The value
must remain between `0 V` and `0.50 V`.

Spectre output can also be exported to CSV independently:

```bash
python3 import_data.py temp/out.raw --csv temp/out.csv
```
