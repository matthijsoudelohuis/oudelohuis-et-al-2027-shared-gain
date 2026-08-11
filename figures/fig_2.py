# -*- coding: utf-8 -*-
"""
Figure 2 -- Nonlinear transfer functions link additive population input to heterogeneous output modulation

Nonlinear TF fitted to single neurons explains the heterogeneity mechanistically; population rate is the best input variable.

Planned panels
--------------
a. Neurons receive additive population input; nonlinear transfer functions convert this into heterogeneous spiking output
b. Neurons are not categorically multiplicative or additive -- this depends on tuning, operating point, and coupling
c. Fitted transfer function shapes (power law, sigmoid) per neuron
d. Neurons near threshold -> supralinear amplification of tuned inputs -> multiplicative output
e. Saturating neurons -> compressed modulation at preferred stimulus
f. Untuned neurons -> additive-appearing output regardless of coupling
g. Population coupling during spontaneous activity predicts gain better than stimulus-evoked population coupling

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
    """Build Figure 2 at final print size (183 x 247 mm, full page).

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
    save_pdf(fig, os.path.join(FIGDIR, 'Figure_2.pdf'))
