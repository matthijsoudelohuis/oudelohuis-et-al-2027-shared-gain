# -*- coding: utf-8 -*-
"""
Shared helpers for building publication-ready figure PDFs.

Implements the print specification documented in CLAUDE.md ("Publication figure pipeline"):
final print size in mm, 300 dpi, Arial, editable vector text, 0.5-1.5 pt lines, and
reading-order (left-to-right, top-to-bottom) capital-letter panel labels.

Every fig_*.py / fig_s*.py script under figures/ should build its canvas through the
functions here so sizing and labeling stay consistent across the whole figure set.
"""

import os

import matplotlib.pyplot as plt

MM_PER_INCH = 25.4

# Cell Press / Nature-style page and column dimensions (mm).
PAGE_WIDTH_MM = 183
PAGE_HEIGHT_MM = 247
COLUMN_WIDTHS_MM = {
    1: 85,    # single column
    1.5: 114,  # 1.5 columns
    2: 174,   # full page width (two-column layout)
}


def mm2inch(mm):
    """Convert millimeters to inches (matplotlib figure sizes are always in inches)."""
    return mm / MM_PER_INCH

def new_page_figure(width_mm=PAGE_WIDTH_MM, height_mm=PAGE_HEIGHT_MM):
    """Create a blank figure canvas sized exactly at final print dimensions (mm)."""
    return plt.figure(figsize=(mm2inch(width_mm), mm2inch(height_mm)))

def add_panel_label(ax, letter, fontsize=8):
    """Add a bold capital panel label just outside the top-left corner of `ax`."""
    ax.text(-0.12, 1.06, letter, transform=ax.transAxes,
             fontsize=fontsize, fontweight='bold', va='bottom', ha='left')

def add_top_row_panels(fig, letters=('a', 'b', 'c', 'd'), panel_size_mm=32, pitch_mm=40,
                        top_margin_mm=15, page_width_mm=PAGE_WIDTH_MM, placeholder=True):
    """Place square placeholder panels along the top row of `fig`.

    Panels are `pitch_mm` apart (left-edge to left-edge), centered on `page_width_mm`, and
    each labeled with its panel letter via add_panel_label. Returns the axes in the same
    order as `letters`, ready to be replaced with real panel content.
    """
    n = len(letters)
    row_width_mm = (n - 1) * pitch_mm + panel_size_mm
    left_margin_mm = (page_width_mm - row_width_mm) / 2

    fig_width_mm = fig.get_size_inches()[0] * MM_PER_INCH
    fig_height_mm = fig.get_size_inches()[1] * MM_PER_INCH

    axes = []
    for i, letter in enumerate(letters):
        left_mm = left_margin_mm + i * pitch_mm
        width_frac = panel_size_mm / fig_width_mm
        height_frac = panel_size_mm / fig_height_mm
        left_frac = left_mm / fig_width_mm
        bottom_frac = 1 - (top_margin_mm / fig_height_mm) - height_frac

        ax = fig.add_axes([left_frac, bottom_frac, width_frac, height_frac])
        add_panel_label(ax, letter)

        if placeholder:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
            ax.text(0.5, 0.5, f'panel {letter}', ha='center', va='center',
                     fontsize=6, color='0.6', transform=ax.transAxes)

        axes.append(ax)
    return axes


def save_pdf(fig, path):
    """Save `fig` as a vector PDF with editable text, at exactly its current figure size.

    Deliberately does not use bbox_inches='tight': that would resize the canvas away from
    the intended print dimensions.
    """
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(path, format='pdf')
    plt.close(fig)
    return path
