# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Analysis code for a systems-neuroscience paper (Oude Lohuis et al.) studying shared population gain
in mouse visual cortex (V1/PM) using two-photon calcium imaging. The repo is a collection of
numbered analysis scripts (`1_scripts/`), all built on top of one importable library package
(`sharedgain/`: data loading in `sharedgain.loaddata`, shared helpers in `sharedgain.utils`). It is a
research codebase, not an application: there is no test suite, linter, or CLI entry point — scripts
are run interactively cell-by-cell.

## Setup and environment

- Environment: conda, defined in `env_trimmed.yml` (name `sharedgain`). Create with
  `conda env create -f env_trimmed.yml`.
- Only the library is installed (editable, `pip install -e .`, already included in the yml);
  `pyproject.toml` includes just `sharedgain*`. Scripts in `1_scripts/` are run, never imported
  (folder names starting with a digit are not importable anyway). All intra-repo imports are
  absolute: `from sharedgain.loaddata.session_info import filter_sessions`,
  `from sharedgain.utils.plot_lib import *`, etc.
- No build, lint, or test commands exist. Do not invent `pytest`/`ruff`/etc. invocations — verify
  changes by running the relevant script or importing the module.
- Scripts use `#%%` cell markers and are meant to be run interactively (VSCode/Spyder Jupyter-style
  cells), not as top-to-bottom CLI scripts. Preserve this style when editing analysis scripts.

## Data access (machine-dependent — read before touching `sharedgain/loaddata/get_data_folder.py`)

Raw/processed data is **not** in the repo; it lives on machine-specific drive letters resolved at
runtime from `os.environ['USERDOMAIN']`:

- `get_data_folder()` and `get_local_drive()` in `sharedgain/loaddata/get_data_folder.py` hard-code a
  drive path per known `USERDOMAIN` (e.g. `MATTHIJSOUDELOH` → `V:/Procdata`, `PCMatthijs` →
  `E:/Procdata`). Running on a new machine requires adding a new `elif` branch there — this is
  expected and not something to "fix" or refactor away.
- `get_rawdata_drive()` / `get_animals_protocol()` map individual animal IDs to drive letters and to
  the protocol family they belong to (`IM/GR/GN/SP/RF` vs `VR/DM/DN/DP`).
- Session data on disk is organized as `<DATA_FOLDER>/<protocol>/<animal_id>/<sessiondate>/` containing
  CSVs: `sessiondata.csv`, `trialdata.csv`, `celldata.csv`, `behaviordata.csv`, `videodata.csv`,
  `dFdata.csv`/`deconvdata.csv`, `Ftsdata.csv`, `Fchan2data.csv`.

## Core architecture: the `Session` object

Nearly everything in this repo flows through `sharedgain/loaddata/session.py`'s `Session` class:

- `Session(protocol, animal_id, sessiondate)` just resolves paths; no I/O happens until `load_data()`
  is called (shallow load by default: sessiondata/trialdata/celldata only).
- `load_data(load_behaviordata, load_calciumdata, load_videodata, calciumversion='dF'|'deconv', filter_hp=...)`
  reads the CSVs above, applies `self.cellfilter` (a boolean mask over cells, e.g. from area
  filtering) if set, and interpolates behavior onto the imaging timebase (`ts_F`) when both calcium
  and behavior are loaded.
- `load_respmat(...)` calls `load_data` then collapses each trial to a single mean response per cell
  over a protocol-specific time window (different `t_resp_start/stop` for `IM`/`GR`/`GN`), producing
  `respmat` (neurons × trials) plus matched `respmat_runspeed`, `respmat_videome`, `respmat_videopc`,
  and pupil measures when available. By default drops the raw traces afterward (`keepraw=False`).
- `load_tensor(...)` is the analogous 3D version: neurons × trials × time (`tensor`, `t_axis`),
  windowed per protocol (`t_pre`/`t_post`).
- Session filtering/batch loading lives in `sharedgain/loaddata/session_info.py`: `filter_sessions(protocols, ...)`
  is the standard entry point used at the top of almost every analysis script to get a list of
  `Session` objects matching criteria (min cells, min trials, specific areas, pupil availability,
  etc.); `load_sessions(protocol, session_list)` loads an explicit `[[animal_id, sessiondate], ...]`
  array.

Typical script pattern: `sessions, nSessions = filter_sessions(protocols=..., ...)` →
`sessions[i].load_data(...)` / `load_respmat(...)` in a loop → concatenate `celldata`/`respmat` across
sessions → analyze/plot.

## Repository layout

```
1_scripts/<NN_step>/<NNx_name>.py   analysis scripts, numbered by step of the analysis story
2_pipeline/<NNx_name>/{out,store}   one folder per script, same name as the script
3_output/figures/                   finished publication figures (from 1_scripts/99_figures)
sharedgain/                         the only importable package (library code)
```

### `1_scripts/` — numbered analysis steps

The two-digit step number gives the reading/dependency order; the letter orders scripts within a step.
Number by analysis order, not by paper figure number — figures get reshuffled across revisions,
step numbers should not.

- `00_datasets/` — `00a_create_datasets.py`: builds the cross-dataset `.npz` files (A–E) that many
  later scripts load via `pipeline_dir('00a_create_datasets','store')`.
- `01_poprate/` — population rate, locomotion gain.
- `02_popcoupling/` — population coupling vs. spontaneous/GR gain, tuning, drift, RF (`02d_rf_vs_gain`).
- `03_nonlinearTF/` — nonlinear transfer-function fits (`03a_run_fit_nonlin` → `03b_analyze_fit_nonlin`),
  power law, population-coupling nonlinearity model.
- `04_overdispersion/` — Fano factor / overdispersion vs. coupling.
- `05_multadd/` — affine (multiplicative/additive) model fits, incl. Xia et al. model analysis.
- `06_noisecorrelations/` — noise-correlation structure vs. gain/coupling, spatial, collinear, RF.
- `07_popgeometry/` — population geometry (cone), nonlinear slices.
- `08_interarea/` — inter-area coupling, CCA/PCA between areas.
- `09_species/` — cross-species (macaque) comparison and CCA.
- `10_decoding/` — population decoding vs. gain, Fisher information.
- `11_naturalimages/` — natural-image (IM) protocol: RF mapping, LNP fits, gain/decoding.
- `99_figures/` — publication figure composition (see below).

New scripts get the next free letter in their step (`06q_...`); a new step gets the next two-digit
number. Within a step, long computations and their analysis are separate scripts in sequence, e.g.
`03a_run_fit_nonlin.py` → `03b_analyze_fit_nonlin.py`.

### `2_pipeline/` — per-script outputs

Every script writes **only** into its own folder, named exactly like the script, via
`sharedgain.paths.pipeline_dir`:

```python
from sharedgain.paths import pipeline_dir
figdir    = pipeline_dir('06a_GR_noisecorr', 'out')     # figures, small tables (png/csv are committed)
resultdir = pipeline_dir('03a_run_fit_nonlin', 'store') # model fits, datasets, caches (gitignored)
```

Pass the script name as a string (not `__file__`, which is unreliable in interactive cells). A script
that consumes another script's results reads them from *that* script's folder (e.g. `03b` reads
`pipeline_dir('03a_run_fit_nonlin','store')`). Never write outputs to OneDrive or hard-coded drive
paths; `params` no longer has `figdir`/`resultdir`.

### `sharedgain/` — library

- `sharedgain/loaddata/` — the `Session` class and session filtering/loading (see above). Change with
  care since every script depends on it.
- `sharedgain/utils/` — shared analysis/plotting helpers: `psth.py` (response-matrix and tensor
  computation used by `Session`), `gain_lib.py`, `rf_lib.py` (receptive fields), `pair_lib.py`
  (pairwise/anatomical distance), `corr_lib.py`/`CCAlib.py` (correlations, CCA), `regress_lib.py`,
  `nonlin_lib.py`, `shuffle_lib.py`, `plot_lib.py` (shared color schemes/plot styling — import `*`
  from this for consistent figure style), `tuning.py`, `arrayop_lib.py`, `imagelib.py`,
  `explorefigs.py`, `params.py`, `fct_data.py`.
- `sharedgain/fct_stat/` — statistical model code ported from Xia et al. (factor-analysis / affine
  models, PyTorch).
- `sharedgain/plots/` — promoted, final panel functions, one module per topic (see below).
- `sharedgain/figlib.py` — publication figure canvas, panel labels, PDF export.
- `sharedgain/paths.py` — `pipeline_dir`, `output_dir`, `datasets_dir`.
- `sharedgain/data/` — small static data used by the library (natural-image stimulus set).

## Publication figure pipeline (`1_scripts/99_figures/`)

`1_scripts/99_figures/` produces the final, publication-ready main and supplementary figures. It sits
on top of the numbered steps (reads their `2_pipeline` results, calls `sharedgain.plots` functions)
rather than duplicating analysis logic, and writes finished PDFs to `3_output/figures/`
(`FIGDIR = output_dir('figures')`).

### Script naming and separation of concerns

- **`NNx_run_xxx.py`** — long-running computation only (model fits, permutation/shuffle tests, CCA
  over many sessions, etc.). No plotting. Saves its output (arrays/dataframes, e.g. pickle/`.npz`/`.csv`)
  to its `2_pipeline/<script>/store/` so it only needs to be re-run when the underlying analysis
  changes, not every time a figure is restyled.
- **`NNy_ana_xxx.py`** (or `analyze_xxx`) — the next letter in the same step: loads the saved output of
  the matching run script, does the (fast) statistics/aggregation needed for plotting, and saves tidy
  results. No long recomputation here — if it needs the run script's output and that output is
  missing, it should say so rather than silently recomputing.
- **`fig_<n>.py`** / **`fig_s<n>.py`** — one script per main or supplementary figure of the paper
  (e.g. `fig_2.py`, `fig_s3.py`). Each script: calls the relevant `ana_xxx` functions for its panels,
  builds each panel, composes them into a single figure at final print dimensions, adds panel labels,
  and exports the finished PDF. A figure script should not itself contain long-running computation —
  that belongs in `run_xxx`/`ana_xxx` upstream.

### Panel functions: keep exploration free, promote only once a panel is final

Manuscript figures get reorganized across revisions (a panel drawn for a main figure in v1 may
move to a supplementary figure in v3, or vice versa). To make that a one-line change instead of a
rewrite, panel-drawing code and figure-composition code stay in different places:

- **Explore freely, unwrapped, in the numbered `#%%` script.** Don't wrap a panel in a function while
  still developing it — work directly on local variables in the interactive kernel (VSCode
  Python Interactive / Jupyter), exactly as today. Wrapping too early just adds a function
  signature you have to keep editing while the plot itself is still changing.
- **Promote to a function only once the panel is done.** Once a cell produces the panel you want,
  copy that cell's body into a `plot_xxx(ax, data, **kwargs)` function in a side-effect-free topic
  module under `sharedgain/plots/` (e.g. `sharedgain/plots/noisecorrelations.py`), analogous to
  `sharedgain/utils/plot_lib.py`. This module must contain **only function definitions** — no
  top-level data loading or computation — because `99_figures/fig_*.py` scripts import from it
  directly; a `#%%` exploration script can't (and shouldn't) be imported, since that would re-run all
  of its top-level loading code.
  Keep the function signature narrow and stable: one pre-built data object (DataFrame/array) plus
  `ax`, not a growing list of loose variables, so later panel reassignment doesn't also force a
  signature change.
- **Get interactivity back after promotion with autoreload.** `%load_ext autoreload` +
  `%autoreload 2` in the interactive kernel picks up edits to `sharedgain/plots/*.py` on the next
  cell run — no kernel restart — so a promoted panel function can still be tweaked live from the
  numbered script's `#%%` cells.
- **`99_figures/fig_<n>.py` only composes:** it creates axes at the desired grid position (via
  `sharedgain/figlib.py`) and calls the already-promoted `plot_xxx(ax, data)` functions into them —
  no plotting logic lives in the `fig_*.py` scripts themselves. Moving a panel from one figure to
  another is then just moving its `plot_xxx(ax, ...)` call from one `fig_*.py` file to another.
- **Standalone export for PowerPoint/Illustrator during exploration:** save any panel worth keeping
  as both PDF (vector, editable in Illustrator/Affinity) and PNG (for quick use in slides) via a
  shared `save_panel(fig, name)` helper in `sharedgain/figlib.py`, independent of whether/where it
  ends up composed into a final figure.

### Print specification (Cell Press / Nature-style guidelines)

Build every figure at **final print size**, not scaled up in Illustrator/Affinity afterward:

- Page maximum: 183 mm × 247 mm (A4).
- Standard widths: 85 mm (1 column), 114 mm (1.5 columns), 174 mm (full page width).
- Export at 300 dpi at the final print size for any rasterized content (images, heatmaps, dense
  scatter). Line art stays vector.
- Export as **PDF with editable text/vector paths** (not text converted to outlines, not the whole
  panel rasterized) so panels remain editable in Illustrator/Affinity Designer 2. In matplotlib this
  means `matplotlib.rcParams['pdf.fonttype'] = 42` (TrueType, keeps text as text) and
  `fig.savefig(path, format='pdf', dpi=300)` — do not use `bbox_inches='tight'` for the final export,
  since it silently changes the canvas size away from the target print dimensions; set the figure
  size in inches (`mm / 25.4`) explicitly instead.
- Font: Arial only, sized 6–8 pt *at final print size* (i.e. set in the rcParams/text calls used at
  the final figure size, not scaled from a larger draft). Note: `sharedgain/utils/plot_lib.py`
  currently sets `font.sans-serif: 'DejaVu Sans'` for exploratory analysis plots — figure scripts in
  `99_figures/` should override this to Arial for the final export.
- Colors in RGB. Never use red and green as a contrasting pair in the same panel (not colorblind-safe)
  — prefer the shared palettes in `sharedgain/utils/plot_lib.py`, swapping any red/green contrasts for e.g.
  blue/orange or magenta/green-free alternatives.
- Line weights / stroke widths: 0.5–1.5 pt (matplotlib `linewidth` is already in points, so this maps
  directly — keep `lines.linewidth` and `axes.linewidth` inside this range at final size).
- Keep vertical spacing between panels/subpanels to the minimum needed for legibility — no default
  matplotlib padding left in the final composed figure.
- Panel labels: bold capital letters (a, b, c, ...), placed consistently (e.g. top-left of each
  panel), assigned in reading order — left to right, then top to bottom across the composed figure.
  Use a single shared helper (`add_panel_label` in `sharedgain/figlib.py`) for placing panel labels so numbering and
  styling stay consistent across all `fig_*.py` scripts rather than each figure re-implementing it.

## Conventions to follow when editing analysis scripts

- Protocol codes are short strings, not the full names: `IM` (natural images), `GR` (gratings),
  `GN` (gratings, different variant/noise), `SP` (spontaneous — not trial-based), `RF` (receptive
  field mapping — not trial-based), `VR`/`DM`/`DN`/`DP` (virtual-reality task protocols).
- "Gain" throughout this codebase generally means population/shared gain, typically operationalized
  as each neuron's correlation with the population mean rate (`poprate`), often called
  `popcoupling` in `celldata`.
- Calcium traces come in two versions selected via `calciumversion`: `'dF'` (ΔF/F) or `'deconv'`
  (deconvolved spikes); response-window timing differs between the two and is already handled inside
  `Session` — don't hard-code new windows elsewhere without checking `load_respmat`/`load_tensor`.
