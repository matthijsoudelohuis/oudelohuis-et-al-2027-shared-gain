# -*- coding: utf-8 -*-
"""
Figure 5 -- Interarea shared variability is dominated by population rate and mediated by choristers

Main claim: what has been interpreted as interarea communication reflects predominantly shared population rate fluctuations, carried disproportionately by population-coupled neurons.

Planned panels
--------------
a. Population rate is correlated across different brain areas
b. The interarea correlation structure is predicted by the product of population coupling and TF slopes across areas -- the same framework that predicts within-area noise correlations extends to cross-area
c. Cone geometry is replicated across simultaneously recorded areas V1, PM and AL -- shared gain produces shared geometry
d. First CCA dimension between simultaneously recorded V1 and PM reflects population rate / behavioral state, not stimulus-specific communication
e. Choristers mediate interarea shared variance not merely because they are coupled locally, but because their supralinear operating regime amplifies shared fluctuations disproportionately -- CCA between chorister subpopulations outperforms soloist subpopulations
f. Local population coupling predicts cross-area population coupling -- a neuron that is a chorister locally tends to be coupled to distant areas as well
g. Feedforward vs. feedback neurons differ in gain modulation -- a specific prediction about the directionality of gain-modulated signals

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
    """Build Figure 5 at final print size (183 x 247 mm, full page).

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
    save_pdf(fig, os.path.join(FIGDIR, 'Figure_5.pdf'))
