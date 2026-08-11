# -*- coding: utf-8 -*-
"""
Figure 6 -- Generalization of the mechanistic framework across species

Planned panels
--------------
a. Same framework of nonlinear TF + population coupling applies despite very different circuits, behavioral states, recording conditions
b. Primate data shows multiarea cone geometry as well
c. Cross-species alignment: mouse and anesthetized monkey V1 share the same gain-scaled cone geometry (aligned using CCA)
d. Population rate is a better alignment variable than locomotion, which does not generalize across species -- a mechanistic claim that nonlinear population gain is a circuit-level property across species
e. Choristers show the largest contributions to CCA across areas

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
    """Build Figure 6 at final print size (183 x 247 mm, full page).

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
    save_pdf(fig, os.path.join(FIGDIR, 'Figure_6.pdf'))
