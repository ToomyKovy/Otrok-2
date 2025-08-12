import matplotlib.pyplot as plt
import seaborn as sns
import numbers
import numpy as np

# ── GLOBAL STYLE KNOBS ────────────────────────────────────────────────────
COL_SPACE     = 15      # vertical white gap (px) between column-groups
ROW_SPACE     = 1.5       # thickness of horizontal grid lines (px)
HEADER_OFFSET = -0.4    # vertical offset of group title (data-coords);
                        #   −0.4 ≈ just above the top row.  Make more negative = higher.

col_types = {
    'Business': ['Product Business', 'General Business'],
    'Data': ['Data Quality', 'Open Banking (PSD2 Standard)', 'Scoring – Credit Risk', 'CRM Labelling', 'Solution Replacement (incl. Internal)'],
    'ESG': ['ESG – Scope Reporting', 'ESG – CO2 Footprint'],
    'UX': ['Transaction history', 'Mastercard Mandate', 'PFM', 'Subscription', 'ATM / Withdrawal', 'AI Chatbot'],
}


# ── MAIN PLOTTER ──────────────────────────────────────────────────────────
def plot_persona_heatmap(df, save_path, w=1., h=0.4, show=False):
    """
    Expects columns:  name | col_type | persona columns.
    Persona columns are ordered by `col_types`;
    group headers shown above blocks with configurable spacing.
    """

    if len(df) == 0:
        print("No data to plot.")
        return

    # 1‣ Order persona columns ------------------------------------------------
    persona_cols = [c for grp in col_types.values() for c in sorted(grp)]
    dfv = df.copy()
    dfv["name"] = dfv["name"] + " (" + dfv["original_segment"] + ")"
    for col in persona_cols:
        dfv[col] = dfv[col].apply(lambda x: x if isinstance(x, numbers.Number) else np.nan)
    data = dfv.set_index('name')[persona_cols].astype(float)

    n_rows, n_cols = data.shape
    fig, ax = plt.subplots(figsize=(w * n_cols, h * n_rows))

    # 2‣ Base heat-map --------------------------------------------------------
    sns.heatmap(
        data, ax=ax,
        cmap=sns.color_palette('RdBu_r', as_cmap=True), vmin=0, vmax=5,
        annot=data.fillna(''), fmt='',
        linewidths=ROW_SPACE, linecolor='white',
        cbar_kws={'label': 'Score'}
    )

    # ax.set_xlabel('Persona type')
    ax.set_ylabel('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    # ax.set_title('Persona scores by name', pad=12)

    # 3‣ Group headers & vertical gaps ---------------------------------------
    start = 0
    for grp, cols in col_types.items():
        present = [c for c in cols if c in data.columns]
        if not present:
            continue
        width = len(present)
        end   = start + width

        # header text (y = HEADER_OFFSET rows above the top-row)
        ax.text(start + width / 2, HEADER_OFFSET, grp,
                ha='center', va='center', fontweight='bold',
                fontsize='medium', transform=ax.transData)

        # white gap after block (except last)
        if end < n_cols:
            ax.vlines(end, *ax.get_ylim(), colors='white',
                      linewidth=COL_SPACE, zorder=4)

        start = end

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

    if show:
        plt.tight_layout()
        plt.show()

    plt.close()
