import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Đảm bảo có thể import module src
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.analyze import load_and_clean

@pytest.fixture
def sample_data_path(tmp_path):
    # Tạo dataset khoảng 100 dòng để quantile(0.99) hoạt động ổn định
    n_rows = 100
    
    df = pd.DataFrame({
        "CreditGrade": pd.Series([np.nan] * n_rows, dtype=object),
        "ProsperRating (Alpha)": pd.Series([np.nan] * n_rows, dtype=object),
        "LoanStatus": ["Current"] * n_rows,
        "DebtToIncomeRatio": [0.2] * n_rows,
        "StatedMonthlyIncome": [5000] * n_rows
    })
    
    # Gán các case đặc biệt vào những dòng đầu tiên
    
    # Case cho test_unify_rating
    # Dòng 0: Chỉ có CreditGrade (Pre-2009)
    df.loc[0, "CreditGrade"] = "A"
    # Dòng 1: Chỉ có ProsperRating (Post-2009)
    df.loc[1, "ProsperRating (Alpha)"] = "AA"
    # Dòng 2: Cả 2 đều NaN (Unknown) Giữ nguyên NaN đã khởi tạo
    
    # Case cho test_flag_risky_loan
    df.loc[0, "LoanStatus"] = "Chargedoff"
    df.loc[1, "LoanStatus"] = "Defaulted"
    df.loc[2, "LoanStatus"] = "Current"
    df.loc[3, "LoanStatus"] = "Completed"
    df.loc[4, "LoanStatus"] = np.nan
    
    # Case cho test_cap_outlier_dti
    df.loc[0, "DebtToIncomeRatio"] = 2.5   # Outlier
    df.loc[1, "DebtToIncomeRatio"] = -0.5  # Boundary/invalid case (DTI âm)
    df.loc[2, "DebtToIncomeRatio"] = np.nan # Boundary/invalid case (NaN)
    
    file_path = tmp_path / "mock_data.csv"
    df.to_csv(file_path, index=False)
    return file_path


def test_unify_rating(sample_data_path):
    """
    Kiểm tra hàm hợp nhất CreditGrade/ProsperRating 
    gán đúng nhãn Pre-2009 / Post-2009 / Unknown.
    """
    df_clean = load_and_clean(sample_data_path)
    
    # Row 0: Pre-2009
    assert df_clean.loc[0, "RatingPeriod"] == "Pre-2009 (CreditGrade)"
    assert df_clean.loc[0, "UnifiedRating"] == "A"
    
    # Row 1: Post-2009
    assert df_clean.loc[1, "RatingPeriod"] == "Post-2009 (ProsperRating)"
    assert df_clean.loc[1, "UnifiedRating"] == "AA"
    
    # Row 2: Unknown (Missing both)
    assert df_clean.loc[2, "RatingPeriod"] == "Unknown"
    assert pd.isna(df_clean.loc[2, "UnifiedRating"])


def test_cap_outlier_dti(sample_data_path):
    """
    Kiểm tra hàm cap DebtToIncomeRatio tại percentile 99 
    bao gồm cả boundary/invalid case (âm, NaN).
    """
    df_clean = load_and_clean(sample_data_path)
    
    # Giá trị p99 của cột DebtToIncomeRatio
    p99 = df_clean["DebtToIncomeRatio"].quantile(0.99)
    
    # Row 0: Outlier (2.5 > p99)
    assert df_clean.loc[0, "is_DebtToIncomeRatio_outlier"] == True
    assert df_clean.loc[0, "DebtToIncomeRatio_capped"] == pytest.approx(p99)
    
    # Row 1: DTI âm (Invalid/Boundary) -> vẫn giữ nguyên vì không lớn hơn p99
    assert df_clean.loc[1, "is_DebtToIncomeRatio_outlier"] == False
    assert df_clean.loc[1, "DebtToIncomeRatio_capped"] == -0.5
    
    # Row 2: DTI là NaN (Invalid/Boundary)
    assert df_clean.loc[2, "is_DebtToIncomeRatio_outlier"] == False
    assert pd.isna(df_clean.loc[2, "DebtToIncomeRatio_capped"])
    

def test_flag_risky_loan(sample_data_path):
    """
    Kiểm tra hàm gắn nhãn IsRiskyLoan = 1 đúng cho LoanStatus 
    thuộc {Chargedoff, Defaulted}, = 0 cho các trạng thái còn lại.
    """
    df_clean = load_and_clean(sample_data_path)
    
    # Row 0: Chargedoff -> 1 (Risky)
    assert df_clean.loc[0, "IsRiskyLoan"] == 1
    
    # Row 1: Defaulted -> 1 (Risky)
    assert df_clean.loc[1, "IsRiskyLoan"] == 1
    
    # Row 2: Current -> 0 (Not Risky)
    assert df_clean.loc[2, "IsRiskyLoan"] == 0
    
    # Row 3: Completed -> 0 (Not Risky)
    assert df_clean.loc[3, "IsRiskyLoan"] == 0
    
    # Row 4: NaN -> 0 (Not Risky)
    assert df_clean.loc[4, "IsRiskyLoan"] == 0
