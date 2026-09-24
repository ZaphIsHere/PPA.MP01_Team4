"""
visualizer.py
-------------
Chức năng: tạo và lưu biểu đồ cho 3 hypotheses vào outputs/figures/.
  - H1: box plot (BorrowerRate theo ProsperRating)
  - H2: bar chart (Risk Rate theo IncomeRange)
  - H3: scatter plot + boxplot (DebtToIncomeRatio vs BorrowerRate)

Chạy độc lập:  python src/visualizer.py
Hoặc import:   from src.visualizer import plot_h1_boxplot, plot_h2_bar, ...
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# =====================================================================
# CẤU HÌNH ĐƯỜNG DẪN
# =====================================================================
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "data_clean.csv"
FIGURES_DIR = ROOT / "outputs" / "figures"

# Thứ tự rủi ro tăng dần (AA -> HR)
RATING_ORDER = ["AA", "A", "B", "C", "D", "E", "HR"]

INCOME_ORDER = [
    "$0",
    "Not employed",
    "$1-24,999",
    "$25,000-49,999",
    "$50,000-74,999",
    "$75,000-99,999",
    "$100,000+",
    "Not displayed",
]


# =====================================================================
# H1: Box Plot — BorrowerRate theo ProsperRating
# =====================================================================
def plot_h1_boxplot(df: pd.DataFrame, output_path: Path) -> Path:
    """Vẽ box plot BorrowerRate theo ProsperRating, lưu file PNG."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Lọc mẫu hợp lệ
    valid_mask = df["ProsperRating (Alpha)"].notna() & df["BorrowerRate"].notna()
    df_h1 = df.loc[valid_mask].copy()

    # Tính mean theo nhóm
    means = (
        df_h1.groupby("ProsperRating (Alpha)")["BorrowerRate"]
        .mean()
        .reindex(RATING_ORDER)
        .values
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("Spectral_r", n_colors=len(RATING_ORDER))
    sns.boxplot(
        data=df_h1,
        x="ProsperRating (Alpha)",
        y="BorrowerRate",
        order=RATING_ORDER,
        palette=palette,
        hue="ProsperRating (Alpha)",
        legend=False,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markeredgecolor": "black",
            "markerfacecolor": "yellow",
            "markersize": 7,
        },
        fliersize=2,
        linewidth=1.2,
        ax=ax,
    )

    # Nhãn giá trị trung bình
    for idx, mean_val in enumerate(means):
        ax.text(
            idx, mean_val + 0.015, f"{mean_val * 100:.1f}%",
            ha="center", va="bottom", fontsize=9,
            fontweight="bold", color="#1a1a1a",
        )

    ax.set_title(
        "H1: Borrower Interest Rate Distribution by Prosper Rating Tier",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.set_xlabel(
        "Prosper Rating Tier (Lowest Risk AA -> Highest Risk HR)",
        fontsize=11, fontweight="semibold",
    )
    ax.set_ylabel("Borrower Rate (Decimal / %)", fontsize=11, fontweight="semibold")
    ax.set_ylim(-0.02, 0.42)
    ax.yaxis.set_major_formatter(lambda y, _: f"{(y * 100):.0f}%")
    ax.scatter(
        [], [], marker="D", edgecolor="black", facecolor="yellow",
        s=50, label="Mean Rate",
    )
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()

    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[H1] Da luu figure: {output_path}")
    return output_path


# =====================================================================
# H2: Bar Chart — Risk Rate theo IncomeRange
# =====================================================================
def plot_h2_bar(df: pd.DataFrame, output_path: Path) -> Path:
    """Vẽ bar chart tỷ lệ rủi ro theo IncomeRange, lưu file PNG."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Tạo cờ rủi ro nếu chưa có
    if "IsRiskyLoan" not in df.columns:
        risky_statuses = {"Chargedoff", "Defaulted"}
        df = df.copy()
        df["IsRiskyLoan"] = df["LoanStatus"].isin(risky_statuses).astype(int)

    valid_income = df["IncomeRange"].isin(INCOME_ORDER)
    df_h2 = df.loc[valid_income].copy()

    h2_table = (
        df_h2.groupby("IncomeRange")
        .agg(
            total_loans=("IsRiskyLoan", "count"),
            risky_rate=("IsRiskyLoan", "mean"),
        )
        .reindex(INCOME_ORDER)
        .dropna(subset=["total_loans"])
    )
    h2_table["risky_rate_pct"] = np.round(h2_table["risky_rate"] * 100, 2)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6))

    categories = list(h2_table.index)
    rates = h2_table["risky_rate_pct"].values
    totals = h2_table["total_loans"].values

    colors = [
        "#d95f02" if "Not" in cat or cat == "$0" else "#1f77b4"
        for cat in categories
    ]
    bars = ax.bar(
        categories, rates, color=colors, edgecolor="black",
        linewidth=0.8, alpha=0.85,
    )

    for bar, rate, total in zip(bars, rates, totals):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.8,
            f"{rate:.1f}%\n(N={int(total):,})",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold",
        )

    ax.set_title(
        "H2: Loan Default & Charged-off Rate across Income Range Categories",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.set_xlabel("Income Range Bracket", fontsize=11, fontweight="semibold")
    ax.set_ylabel(
        "Risk Rate (%) [Defaulted + Charged-off]",
        fontsize=11, fontweight="semibold",
    )
    ax.set_ylim(0, max(rates) + 8)
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0f}%")
    plt.xticks(rotation=25, ha="right", fontsize=9.5)

    fig.text(
        0.5, -0.06,
        "* Note: Visual association does not imply causality (FR07). "
        "Sample sizes annotated as N.",
        ha="center", fontsize=9, style="italic", color="#555555",
    )
    plt.tight_layout()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[H2] Da luu figure: {output_path}")
    return output_path


# =====================================================================
# H3: Scatter Plot + Boxplot — DebtToIncomeRatio vs BorrowerRate
# =====================================================================
def plot_h3_scatter(df: pd.DataFrame, output_path: Path) -> Path:
    """Vẽ scatter plot DebtToIncomeRatio vs BorrowerRate kèm trendline."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    col_x = "DebtToIncomeRatio_capped" if "DebtToIncomeRatio_capped" in df.columns else "DebtToIncomeRatio"
    col_y = "BorrowerRate"

    df_h3 = df[[col_x, col_y]].dropna().reset_index(drop=True)
    res = stats.linregress(df_h3[col_x], df_h3[col_y])

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(
        x=col_x, y=col_y, data=df_h3, ax=ax,
        scatter_kws={"alpha": 0.1, "color": "#4C72B0", "s": 15},
        line_kws={"color": "red", "linewidth": 2},
    )
    ax.set_title(
        "H3: DebtToIncomeRatio vs BorrowerRate (Scatter + Trendline)",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.set_xlabel(
        "Ty le no tren thu nhap (DebtToIncomeRatio, capped tai P99)",
        fontsize=12,
    )
    ax.set_ylabel("Lai suat khoan vay (BorrowerRate)", fontsize=12)
    ax.text(
        0.02, 0.97,
        f"n = {len(df_h3):,}\nPearson r = {res.rvalue:+.3f}\n"
        f"R2 = {res.rvalue ** 2:.3f}",
        transform=ax.transAxes, va="top", fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"),
    )
    fig.tight_layout()

    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[H3] Da luu figure: {output_path}")
    return output_path


def plot_h3_boxplot(df: pd.DataFrame, output_path: Path) -> Path:
    """Vẽ boxplot BorrowerRate theo tứ phân vị DebtToIncomeRatio."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    col_x = "DebtToIncomeRatio_capped" if "DebtToIncomeRatio_capped" in df.columns else "DebtToIncomeRatio"
    col_y = "BorrowerRate"

    df_h3 = df[[col_x, col_y]].dropna().reset_index(drop=True)

    # Tạo tứ phân vị
    q = pd.qcut(df_h3[col_x], 4, duplicates="drop")
    edges = q.cat.categories
    labels = [
        f"Q{i + 1}\n(<= {iv.right:.2f})" if i == 0
        else f"Q{i + 1}\n({iv.left:.2f}-{iv.right:.2f})"
        for i, iv in enumerate(edges)
    ]
    df_h3["DTI_quartile"] = pd.qcut(
        df_h3[col_x], 4, labels=labels, duplicates="drop"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x="DTI_quartile", y=col_y, data=df_h3, ax=ax, color="#4C72B0")
    ax.set_title(
        "H3: BorrowerRate theo tu phan vi DebtToIncomeRatio",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.set_xlabel("Tu phan vi DTI", fontsize=12)
    ax.set_ylabel("Lai suat khoan vay (BorrowerRate)", fontsize=12)
    fig.tight_layout()

    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[H3] Da luu figure: {output_path}")
    return output_path


# =====================================================================
# MAIN — chạy độc lập để tạo tất cả figures
# =====================================================================
def main() -> None:
    """Đọc data và tạo tất cả biểu đồ cho 3 hypotheses."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[Visualizer] Dang doc du lieu: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"[Visualizer] Da doc: {len(df):,} dong x {df.shape[1]} cot")

    # Tạo cờ rủi ro nếu chưa có
    if "IsRiskyLoan" not in df.columns:
        risky_statuses = {"Chargedoff", "Defaulted"}
        df["IsRiskyLoan"] = df["LoanStatus"].isin(risky_statuses).astype(int)

    # Cap outlier nếu chưa có cột capped
    if "DebtToIncomeRatio_capped" not in df.columns and "DebtToIncomeRatio" in df.columns:
        p99 = df["DebtToIncomeRatio"].quantile(0.99)
        df["DebtToIncomeRatio_capped"] = df["DebtToIncomeRatio"].clip(upper=p99)

    # Vẽ tất cả biểu đồ
    plot_h1_boxplot(df, FIGURES_DIR / "h1_borrower_rate_by_rating.png")
    plot_h2_bar(df, FIGURES_DIR / "h2_risk_rate_by_income.png")
    plot_h3_scatter(df, FIGURES_DIR / "h3_scatter_dti_rate.png")
    plot_h3_boxplot(df, FIGURES_DIR / "h3_boxplot_dti_quartile.png")

    print(f"\n[Visualizer] Hoan tat. Tat ca figures da luu tai: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
