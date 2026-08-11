# -*- coding: utf-8 -*-
"""
Figure 3 -- Heterogeneous gain modulation explains tuning-dependent noise correlations

With the mechanistic model established, the first population-level consequence follows.

Planned panels
--------------
a. Noise correlations are tuning-dependent
b. Framework predicts: NC(i,j,k) proportional to gamma_i * gamma_j * f'(u_i(k)) * f'(u_j(k))
c. Noise correlations should be stimulus dependent and not averaged across neurons
d. Prediction validated: product of population coupling and TF-slope product explains per-stimulus noise correlations
e. Nonlinear TF outperforms linear model; explains hockey-stick relationship between signal and noise correlations
f. Cross-session demonstration: shared gain alone -- without direct connectivity -- generates tuning-dependent noise correlations
g. Like-to-like tuning-dependent correlations locally during spontaneous activity (shared gain does not explain all tuning-dependent correlations)

This script only builds the print-ready figure canvas and panel scaffold (see
figures/fig_utils.py and CLAUDE.md, "Publication figure pipeline"). No analysis or plotting
of real results happens here -- panel content should be filled in from the corresponding
run_xxx / ana_xxx outputs once those analyses are ready.
"""

import os
from figures.fig_utils import new_page_figure, add_top_row_panels, save_pdf

# Figure output directory. Override by editing this path (or reassigning FIGDIR before calling
# make_figure()/save_pdf if importing this module elsewhere).
FIGDIR = r"E:\Documents\Manuscripts\2026 - Heterogeneous gain\v1"


def make_figure():
    """Build Figure 3 at final print size (183 x 247 mm, full page).

    Currently just a placeholder scaffold: four top-row panels (a-d), spaced 40 mm apart,
    labeled and ready to be replaced with real panel content. Additional rows of panels
    should be added below as the figure is fleshed out.
    """
    fig = new_page_figure()
    axes = add_top_row_panels(fig, letters=('a', 'b', 'c', 'd'))
    # TODO: replace placeholder axes[i] content with real panels per the plan above.
    return fig


if __name__ == '__main__':
    fig = make_figure()
    save_pdf(fig, os.path.join(FIGDIR, 'Figure_3.pdf'))
