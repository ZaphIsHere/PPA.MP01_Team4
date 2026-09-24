"""
analyze.py
----------
Chỉ thực hiện phân tích dữ liệu và xuất kết quả CSV.
Không vẽ biểu đồ — việc đó thuộc về visualizer.py.

Output:
  - outputs/hypothesis_results.csv
  - outputs/tables/h1_summary_table.csv
  - outputs/tables/h2_summary_table.csv
  - outputs/tables/h3_correlation_table.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# =====================================================================
# CẤU HÌNH ĐƯỜNG DẪN
# =====================================================================
ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT / "data" / "data_clean.csv"
TABLES_DIR = ROOT / "outputs" / "tables"
RESULTS_PATH = ROOT / "outputs" / "hypothesis_results.csv"

# Tạo thư mục nếu chưa có
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Thứ tự Prosper Rating từ rủi ro thấp → cao
RATING_ORDER = ["AA", "A", "B", "C", "D", "E", "HR"]

# Thứ tự Income Range
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

# Danh sách lưu kết quả tổng hợp của 3 hypothesis
results = []


# =====================================================================
# HÀM ĐỌC VÀ LÀM SẠCH DỮ LIỆU
# =====================================================================
def load_and_clean(path: Path) -> pd.DataFrame:
    """Đọc dữ liệu thô và thực hiện các bước làm sạch cơ bản."""
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu: {path}\n"
            f"Hãy đặt file CSV gốc vào: {path}"
        )

    print(f"[Data] Đang đọc: {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"[Data] Đã đọc: {len(df):,} dòng × {df.shape[1]} cột")

    # Hợp nhất 2 hệ thống rating (Pre-2009 / Post-2009)
    df["RatingPeriod"] = "Unknown"
    df.loc[df["CreditGrade"].notna(), "RatingPeriod"] = "Pre-2009 (CreditGrade)"
    df.loc[
        df["ProsperRating (Alpha)"].notna(), "RatingPeriod"
    ] = "Post-2009 (ProsperRating)"
    df["UnifiedRating"] = df["ProsperRating (Alpha)"].fillna(df["CreditGrade"])

    # Cap outlier tại percentile 99
    for col in ["StatedMonthlyIncome", "DebtToIncomeRatio"]:
        if col in df.columns:
            p99 = df[col].quantile(0.99)
            df[f"is_{col}_outlier"] = df[col] > p99
            df[f"{col}_capped"] = df[col].clip(upper=p99)

    # Cờ rủi ro cho Hypothesis 2
    risky_statuses = {"Chargedoff", "Defaulted"}
    df["IsRiskyLoan"] = df["LoanStatus"].isin(risky_statuses).astype(int)

    print(f"[Data] Làm sạch xong. IsRiskyLoan = {df['IsRiskyLoan'].sum():,} khoản")
    return df


# =====================================================================
# HYPOTHESIS 1: BorrowerRate theo ProsperRating (AA → HR)
# =====================================================================
def run_hypothesis_1(df: pd.DataFrame) -> dict:
    """H1: Lãi suất tăng đơn điệu từ AA đến HR."""
    print("\n" + "=" * 65)
    print("HYPOTHESIS 1: BORROWER RATE THEO PROSPER RATING")
    print("=" * 65)

    # Lọc mẫu có ProsperRating & BorrowerRate
    valid_mask = df["ProsperRating (Alpha)"].notna() & df["BorrowerRate"].notna()
    df_h1 = df.loc[valid_mask].copy()
    n_excluded = len(df) - len(df_h1)

    # Tổng hợp thống kê theo nhóm rating
    grouped = df_h1.groupby("ProsperRating (Alpha)")["BorrowerRate"]
    h1_table = grouped.agg(
        count="count",
        mean=lambda x: np.mean(x),
        std=lambda x: np.std(x, ddof=1),
        median=lambda x: np.median(x),
        q25=lambda x: np.percentile(x, 25),
        q75=lambda x: np.percentile(x, 75),
        min=lambda x: np.min(x),
        max=lambda x: np.max(x),
    ).loc[RATING_ORDER]

    # Tính chênh lệch giữa các bậc liền kề
    means = h1_table["mean"].values
    step_diffs = np.diff(means)
    h1_table["step_diff_pct"] = [0.0] + [round(d * 100, 2) for d in step_diffs]

    # Quy tắc quyết định
    min_step_diff = 0.015
    is_monotonic = bool(np.all(step_diffs > 0))
    all_meet_threshold = bool(np.all(step_diffs >= min_step_diff))
    overall_spread = float(means[-1] - means[0])
    h1_status = "ACCEPTED" if (is_monotonic and all_meet_threshold) else "REJECTED"

    # Lưu bảng
    h1_table.to_csv(TABLES_DIR / "h1_summary_table.csv")

    # In kết quả
    print(f"Tổng mẫu hợp lệ    : {len(df_h1):,} khoản vay")
    print(f"Mẫu bị loại (trước 2009): {n_excluded:,} ({n_excluded / len(df) * 100:.2f}%)")
    print(f"Chênh lệch tổng thể: {overall_spread * 100:.2f} điểm %")
    print(f"Đơn điệu tăng      : {'Có' if is_monotonic else 'Không'}")
    print(f"Tất cả bước >= 1.5%: {'Có' if all_meet_threshold else 'Không'}")
    print(f"Trạng thái          : {h1_status}")
    print("-" * 65)
    print(h1_table[["count", "mean", "median", "step_diff_pct"]])

    return {
        "hypothesis": "H1",
        "description": "BorrowerRate tăng đơn điệu từ AA đến HR",
        "n_sample": int(len(df_h1)),
        "n_excluded": int(n_excluded),
        "is_monotonic": is_monotonic,
        "overall_spread_pct": round(overall_spread * 100, 2),
        "decision": h1_status,
        "table": "outputs/tables/h1_summary_table.csv",
    }


# =====================================================================
# HYPOTHESIS 2: Tỷ lệ rủi ro (IsRiskyLoan) theo IncomeRange
# =====================================================================
def run_hypothesis_2(df: pd.DataFrame) -> dict:
    """H2: Thu nhập thấp → tỷ lệ nợ xấu cao hơn."""
    print("\n" + "=" * 65)
    print("HYPOTHESIS 2: LOAN RISK THEO INCOME RANGE")
    print("=" * 65)

    # Gom nhóm & tính tỷ lệ rủi ro
    valid_income = df["IncomeRange"].isin(INCOME_ORDER)
    df_h2 = df.loc[valid_income].copy()

    h2_table = (
        df_h2.groupby("IncomeRange")
        .agg(
            total_loans=("IsRiskyLoan", "count"),
            risky_loans=("IsRiskyLoan", "sum"),
            risky_rate=("IsRiskyLoan", "mean"),
        )
        .reindex(INCOME_ORDER)
        .dropna(subset=["total_loans"])
    )
    h2_table["risky_rate_pct"] = np.round(h2_table["risky_rate"] * 100, 2)

    # Quy tắc quyết định
    rate_zero = h2_table.loc["$0", "risky_rate_pct"] if "$0" in h2_table.index else None
    rate_high = (
        h2_table.loc["$100,000+", "risky_rate_pct"]
        if "$100,000+" in h2_table.index
        else None
    )
    min_diff_threshold = 5.0

    if rate_zero is not None and rate_high is not None:
        disparity = rate_zero - rate_high
        h2_status = "ACCEPTED" if disparity >= min_diff_threshold else "REJECTED"
    else:
        disparity = None
        h2_status = "INCONCLUSIVE"

    # Lưu bảng
    h2_table.to_csv(TABLES_DIR / "h2_summary_table.csv")

    # In kết quả
    print(f"Tổng khoản vay     : {len(df_h2):,}")
    print(f"Tổng nợ xấu        : {df_h2['IsRiskyLoan'].sum():,} "
          f"({df_h2['IsRiskyLoan'].mean() * 100:.2f}%)")
    if rate_zero is not None:
        print(f"Tỷ lệ rủi ro $0        : {rate_zero:.2f}%")
    if rate_high is not None:
        print(f"Tỷ lệ rủi ro $100,000+ : {rate_high:.2f}%")
    if disparity is not None:
        print(f"Chênh lệch             : {disparity:.2f}% (ngưỡng >= {min_diff_threshold}%)")
    print(f"Trạng thái              : {h2_status}")
    print("-" * 65)
    print(h2_table[["total_loans", "risky_loans", "risky_rate_pct"]])

    return {
        "hypothesis": "H2",
        "description": "Thu nhập thấp có tỷ lệ nợ xấu cao hơn thu nhập cao",
        "n_sample": int(len(df_h2)),
        "n_excluded": int(len(df) - len(df_h2)),
        "rate_zero_pct": float(rate_zero) if rate_zero is not None else None,
        "rate_100k_plus_pct": float(rate_high) if rate_high is not None else None,
        "disparity_pct": float(disparity) if disparity is not None else None,
        "decision": h2_status,
        "table": "outputs/tables/h2_summary_table.csv",
    }


# =====================================================================
# HYPOTHESIS 3: DebtToIncomeRatio vs BorrowerRate
# =====================================================================
def run_hypothesis_3(df: pd.DataFrame) -> dict:
    """H3: DebtToIncomeRatio (capped) có tương quan dương với BorrowerRate."""
    print("\n" + "=" * 65)
    print("HYPOTHESIS 3: DEBTTOINCOME RATIO vs BORROWER RATE")
    print("=" * 65)

    col_x = "DebtToIncomeRatio"
    col_y = "BorrowerRate"

    # Dùng bản capped nếu có, nếu không thì dùng gốc
    if "DebtToIncomeRatio_capped" in df.columns:
        col_x_plot = "DebtToIncomeRatio_capped"
    else:
        col_x_plot = col_x

    # Lọc bỏ missing
    df_h3 = df[[col_x_plot, col_y]].dropna().reset_index(drop=True)
    n_dropped = len(df) - len(df_h3)

    print(f"Dòng ban đầu : {len(df):,}")
    print(f"Bỏ do thiếu  : {n_dropped:,}")
    print(f"Còn lại      : {len(df_h3):,}")

    # Thống kê tương quan
    res = stats.linregress(df_h3[col_x_plot], df_h3[col_y])
    rho, p_rho = stats.spearmanr(df_h3[col_x_plot], df_h3[col_y])

    print(f"Pearson  r   = {res.rvalue:+.3f} (p = {res.pvalue:.3g})")
    print(f"Spearman rho = {rho:+.3f} (p = {p_rho:.3g})")
    print(f"R2           = {res.rvalue ** 2:.3f}")
    print(f"Slope        = {res.slope:.4f} | Intercept = {res.intercept:.4f}")

    # Bảng correlation
    h3_table = pd.DataFrame([{
        "x_col": col_x_plot,
        "y_col": col_y,
        "n_obs": len(df_h3),
        "pearson_r": round(res.rvalue, 4),
        "pearson_p": res.pvalue,
        "spearman_rho": round(rho, 4),
        "spearman_p": p_rho,
        "r_squared": round(res.rvalue ** 2, 4),
        "slope": round(res.slope, 6),
        "intercept": round(res.intercept, 6),
    }])
    h3_table.to_csv(TABLES_DIR / "h3_correlation_table.csv", index=False)

    # Quy tắc quyết định: r > 0 và p < 0.05
    h3_status = (
        "ACCEPTED" if (res.rvalue > 0 and res.pvalue < 0.05) else "REJECTED"
    )
    print(f"Trạng thái   : {h3_status}")

    return {
        "hypothesis": "H3",
        "description": "DebtToIncomeRatio tương quan dương với BorrowerRate",
        "n_sample": int(len(df_h3)),
        "n_excluded": int(n_dropped),
        "pearson_r": round(res.rvalue, 4),
        "pearson_p": res.pvalue,
        "spearman_rho": round(rho, 4),
        "r_squared": round(res.rvalue ** 2, 4),
        "decision": h3_status,
        "table": "outputs/tables/h3_correlation_table.csv",
    }


# =====================================================================
# MAIN
# =====================================================================
def main() -> None:
    """Chạy toàn bộ pipeline phân tích 3 hypotheses."""

    # 1) Đọc và làm sạch dữ liệu
    df = load_and_clean(RAW_DATA_PATH)

    # 2) Chạy 3 hypothesis
    h1_result = run_hypothesis_1(df)
    results.append(h1_result)

    h2_result = run_hypothesis_2(df)
    results.append(h2_result)

    h3_result = run_hypothesis_3(df)
    results.append(h3_result)

    # 3) Xuất bảng tổng hợp kết quả
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)

    # 4) In tổng kết
    print("\n" + "=" * 65)
    print("TONG KET KET QUA 3 HYPOTHESES")
    print("=" * 65)
    for r in results:
        status = r["decision"]
        print(f"  {r['hypothesis']}: {status} -- {r['description']}")
    print("\nHoan tat phan tich.")


if __name__ == "__main__":
    main()
