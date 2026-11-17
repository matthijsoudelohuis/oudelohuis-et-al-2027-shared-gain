# -*- coding: utf-8 -*-
"""
Repository-anchored output locations.

Every analysis script in 1_scripts/ writes only into its own folder under 2_pipeline/, named after the
script (same stem), with two subfolders:
    out/    figures and small tables meant to be looked at (png/csv are committed to git)
    store/  large or regenerable results: model fits, datasets, caches (gitignored)

Publication figures (1_scripts/99_figures/fig_*.py) write to 3_output/ instead.

Usage in a script:
    from sharedgain.paths import pipeline_dir
    figdir = pipeline_dir('06a_GR_noisecorr', 'out')
    savedir = pipeline_dir('03a_run_fit_nonlin', 'store')

The script name is passed as a string rather than derived from __file__, because __file__ is not
reliably defined when running #%% cells interactively.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = REPO_ROOT / '2_pipeline'
OUTPUT_ROOT = REPO_ROOT / '3_output'


def _make(path, create):
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return str(path)


def pipeline_dir(script, *subdirs, create=True):
    """Return 2_pipeline/<script>/<subdirs...> as a string, creating it by default.

    `script` is the script stem (e.g. '03a_run_fit_nonlin'); a path or filename is also accepted.
    """
    return _make(PIPELINE_ROOT.joinpath(Path(script).stem, *subdirs), create)


def output_dir(*subdirs, create=True):
    """Return 3_output/<subdirs...> as a string, creating it by default."""
    return _make(OUTPUT_ROOT.joinpath(*subdirs), create)


def datasets_dir(create=True):
    """Folder holding the .npz datasets written by 1_scripts/00_datasets/00a_create_datasets.py."""
    return pipeline_dir('00a_create_datasets', 'store', create=create)
