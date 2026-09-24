"""
data_loader.py
---------------
Chức năng duy nhất: đọc dữ liệu thô (raw) và cung cấp thông tin profiling
cơ bản (số dòng/cột, kiểu dữ liệu). Không thực hiện bất kỳ bước làm sạch
nào ở đây (việc đó thuộc về data_cleaner.py) — TR02: mỗi module một
mục đích rõ ràng.
"""

from pathlib import Path
import pandas as pd


def load_raw_data(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {path}")
    return pd.read_csv(path, low_memory=False)


def get_basic_profile(df: pd.DataFrame) -> dict:

    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "n_duplicates": int(df.duplicated().sum()),
    }


def classify_columns(df: pd.DataFrame, datetime_cols: list[str] | None = None) -> dict:

    datetime_cols = datetime_cols or []
    numeric_cols = [c for c in df.select_dtypes(include="number").columns]
    categorical_cols = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if c not in datetime_cols
    ]
    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "datetime": [c for c in datetime_cols if c in df.columns],
    }
