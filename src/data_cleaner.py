
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "raw" / "data.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "data_clean.csv"
TRACE_PATH = ROOT / "outputs" / "tables" / "cleaning_trace_log.csv"

trace = []  # data-quality trace log: (bước, số dòng trước, số dòng sau, ghi chú)


def log(step, before, after, note):
    trace.append({"step": step, "rows_before": before, "rows_after": after, "note": note})


def main():
    if not INPUT_PATH.exists():
        print(f"KHÔNG TÌM THẤY FILE: {INPUT_PATH}")
        print("   -> Kiểm tra lại đường dẫn data/raw/data.csv trong project root.")
        return

    df = pd.read_csv(INPUT_PATH, low_memory=False)
    n0 = len(df)
    log("00_load", n0, n0, f"Load gốc: {n0} dòng x {df.shape[1]} cột")
    print(f"Đã đọc file: {n0} dòng x {df.shape[1]} cột")

    # Kiểm tra trùng lặp
    dup_count = df.duplicated().sum()
    log("01_duplicates", len(df), len(df), f"Số dòng trùng lặp hoàn toàn: {dup_count} (không xóa)")

   
    # Hợp nhất CreditGrade / ProsperRating (2 giai đoạn thời gian)
    df["RatingPeriod"] = "Unknown"
    df.loc[df["CreditGrade"].notna(), "RatingPeriod"] = "Pre-2009 (CreditGrade)"
    df.loc[df["ProsperRating (Alpha)"].notna(), "RatingPeriod"] = "Post-2009 (ProsperRating)"

    df["UnifiedRating"] = df["ProsperRating (Alpha)"].fillna(df["CreditGrade"])

    n_pre = (df["RatingPeriod"] == "Pre-2009 (CreditGrade)").sum()
    n_post = (df["RatingPeriod"] == "Post-2009 (ProsperRating)").sum()
    n_unknown = (df["RatingPeriod"] == "Unknown").sum()
    log(
        "02_unify_rating",
        len(df),
        len(df),
        f"Tạo UnifiedRating + RatingPeriod. Pre-2009: {n_pre}, Post-2009: {n_post}, "
        f"Unknown (cả 2 đều thiếu): {n_unknown}",
    )

    # Đánh dấu outlier StatedMonthlyIncome & DebtToIncomeRatio
    # (cap tại percentile 99, KHÔNG xóa dòng — chỉ gắn cờ + tạo bản capped)

    income_p99 = df["StatedMonthlyIncome"].quantile(0.99)
    df["is_income_outlier"] = df["StatedMonthlyIncome"] > income_p99
    df["StatedMonthlyIncome_capped"] = df["StatedMonthlyIncome"].clip(upper=income_p99)
    n_income_outlier = df["is_income_outlier"].sum()

    dti_p99 = df["DebtToIncomeRatio"].quantile(0.99)
    df["is_dti_outlier"] = df["DebtToIncomeRatio"] > dti_p99
    df["DebtToIncomeRatio_capped"] = df["DebtToIncomeRatio"].clip(upper=dti_p99)
    n_dti_outlier = df["is_dti_outlier"].sum()

    log(
        "03_outlier_flag",
        len(df),
        len(df),
        f"StatedMonthlyIncome: cap tại P99={income_p99:.2f}, {n_income_outlier} dòng bị đánh dấu outlier. "
        f"DebtToIncomeRatio: cap tại P99={dti_p99:.4f}, {n_dti_outlier} dòng bị đánh dấu outlier.",
    )


    # Cột rủi ro (risk flag) cho Hypothesis 2
    
    risky_status = {"Chargedoff", "Defaulted"}
    df["IsRiskyLoan"] = df["LoanStatus"].isin(risky_status)
    n_risky = df["IsRiskyLoan"].sum()
    log("04_risk_flag", len(df), len(df), f"Đánh dấu IsRiskyLoan (Chargedoff/Defaulted): {n_risky} dòng")

    
    # Map IncomeRange sang biến ordinal
    
    income_order = {
        "Not employed": 0,
        "$0": 1,
        "$1-24,999": 2,
        "$25,000-49,999": 3,
        "$50,000-74,999": 4,
        "$75,000-99,999": 5,
        "$100,000+": 6,
        "Not displayed": None,
    }
    df["IncomeRange_ordinal"] = df["IncomeRange"].map(income_order)
    n_unmapped = df["IncomeRange_ordinal"].isna().sum() - df["IncomeRange"].isna().sum()
    log(
        "05_income_ordinal",
        len(df),
        len(df),
        f"Map IncomeRange -> IncomeRange_ordinal. Giá trị không map được (ngoài NaN gốc): {n_unmapped}",
    )

    
    # Xuất file + log trace
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    except PermissionError:
        print(f"KHÔNG GHI ĐƯỢC FILE: {OUTPUT_PATH}")
        print("   -> File này có thể đang MỞ trong Excel. Đóng Excel rồi chạy lại.")
        return

    trace_df = pd.DataFrame(trace)
    trace_df.to_csv(TRACE_PATH, index=False, encoding="utf-8-sig")

    print(f"Đã lưu: {OUTPUT_PATH} ({len(df)} dòng x {df.shape[1]} cột — không mất dòng nào)")
    print(f"Đã lưu trace log: {TRACE_PATH}")
    print()
    print(trace_df.to_string(index=False))


if __name__ == "__main__":
    main()
