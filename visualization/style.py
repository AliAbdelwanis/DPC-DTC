import matplotlib.pyplot as plt
def set_plot_style(use_tex: bool = True) -> None:
    # =========================
    # Global Matplotlib + LaTeX settings
    # =========================
    plt.rcParams.update({
        # Use LaTeX for all text
        "text.usetex": False,
        
        # Font family fallback
        #"font.family": "serif",
        #"font.serif": ["Times New Roman"],  # normal text fallback

        # LaTeX preamble
        "text.latex.preamble": r"""
            \usepackage{amsmath}     % math support
            \usepackage{bm}          % bold math
            \usepackage{siunitx}     % SI units
            \usepackage{newtxtext}   % Times-like text
            \usepackage{newtxmath}   % Times-like math
            \sisetup{detect-all,per-mode=symbol}
        """,

        # Lines and markers
        "lines.linewidth": 3,
        "lines.markersize": 15,

        # Font sizes
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,

        # Ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 8,
        "ytick.major.size": 8,
        "xtick.minor.size": 6,
        "ytick.minor.size": 6,
        "xtick.major.width": 1,
        "ytick.major.width": 1,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,

        # Grid
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,

        # Figure background
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "figure.figsize": (10, 6),

        # PDF/PS font embedding (avoid PostScript font errors)
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })