# -*- coding: utf-8 -*-
"""
Supplementary Figure S4 -- supports Figure 4 (Nonlinear population gain modulation produces a characteristic cone geometry of the sensory manifold, whose properties are directly predicted by the gain framework)

The paper outline lists two supplementary blocks for Figure 4; they are combined here into one Supplementary Figure S4.

Planned panels
--------------
a. Cone geometry is independent of locomotion; population rate is the key variable
b. Three conditions for cone geometry: stimulus-tuned neurons + population coupling + population rate fluctuations (each ablation shown)
c. PCA variance decomposition of nonlinear-TF population activity
d. Cone geometry across animals and sessions
e. Cross-session alignment: sorting by stimulus + population rate aligns single-trial representations across animals
f. MultiView CCA procedure

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
    """Build Supplementary Figure S4 at final print size (183 x 247 mm, full page).

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
    save_pdf(fig, os.path.join(FIGDIR, 'Figure_S4.pdf'))
