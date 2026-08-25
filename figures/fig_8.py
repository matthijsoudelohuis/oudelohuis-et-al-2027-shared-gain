# -*- coding: utf-8 -*-
"""
Figure 8 -- Principles generalize to naturalistic visual processing

Planned panels
--------------
a. Multiplicative and additive modulation by population rate observed for natural images (model-free)
b. Tuning-dependent noise correlations replicated, predicted by TF-slope x coupling product
c. Population geometry (cone) observed for natural images as well
d. KNN decoding of image identity improves with population rate and most for intermediate coupling

This script only builds the print-ready figure canvas and panel scaffold (see
figures/fig_utils.py and CLAUDE.md, "Publication figure pipeline"). No analysis or plotting
of real results happens here -- panel content should be filled in from the corresponding
run_xxx / ana_xxx outputs once those analyses are ready.
"""

import os

import numpy as np
from PIL import Image

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None

from figures.fig_utils import new_page_figure, add_top_row_panels, save_pdf
from utils.params import params

# Figure output directory. Override by editing this path (or reassigning FIGDIR before calling
# make_figure()/save_pdf if importing this module elsewhere).
FIGDIR = r"E:\Documents\Manuscripts\2026 - Heterogeneous gain\v1"

def _place_pdf_in_panel(ax, pdf_path):
    """Render the first page of a PDF into a matplotlib axis."""
    if not pdf_path or not os.path.exists(pdf_path):
        return

    if pdf_path.lower().endswith('.pdf') and fitz is not None:
        with fitz.open(pdf_path) as doc:
            if len(doc) == 0:
                return
            page = doc[0]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            image = np.asarray(pix)
            ax.imshow(image)
    else:
        image = Image.open(pdf_path)
        ax.imshow(np.asarray(image))

    ax.set_axis_off()

def make_figure():
    """Build Figure 8 and place the configured PDFs in panels a and b."""
    fig = new_page_figure()
    axes = add_top_row_panels(fig, letters=('a', 'b', 'c', 'd'))

    _place_pdf_in_panel(axes[0], filename_panela)
    _place_pdf_in_panel(axes[1], filename_panelb)

    return fig

figdir = os.path.join(params['figdir'], 'naturalimages', 'decoding')
# filename_panela = os.path.join(figdir, 'KNN_decoding_ActBins.pdf')
filename_panela = os.path.join(figdir, 'KNN_decoding_ActBins.png')
# filename_panelb = os.path.join(figdir, 'KNN_decoding_ActBins_CouplingBins.pdf')
filename_panelb = os.path.join(figdir, 'KNN_decoding_ActBins_CouplingBins.png')

if __name__ == '__main__':
    fig = make_figure()
    save_pdf(fig, os.path.join(FIGDIR, 'Figure_8.pdf'))
