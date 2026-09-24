# PPA.MP01 — Tabular Data Exploration, Analysis, and Visualization

## 1. Thành viên nhóm & vai trò

| Tên               | Vai trò trong dự án 
|-------------------|------------------------------------
| Lưu Quang Thuấn   | Phần 1: Giới thiệu & nguồn dữ liệu 
| Phạm Trung Chiến  | Phần 2: Profiling & cleaning 
| Trương Việt An    | Phần 3: Hypothesis 1 & 2 
| Lê Hải Dương      | Phần 4: Hypothesis 3 & tổng hợp 
| Phan Công Thứ     | Phần 5: Kỹ thuật & kết luận


## 2. Cấu trúc project

```
├── main.py
├── requirements.txt
├── data/
│   ├── raw/prosperLoanData.csv
│   ├── processed/cleaning_trace_log.csv
│   └── dataset_clean.csv
├── src/
│   ├── data_loader.py
│   ├── data_cleaner.py
│   ├── analyzer.py
│   └── visualizer.py
├── tests/
│   └── test_pipeline.py
└── outputs/
    ├── tables/
    └── figures/
```

## 3. Nguồn dữ liệu

- Dataset: Prosper Loan Data
- Kaggle page: (dán link thật)
- License/usage terms: (dán thông tin thật)
- Ngày tải: (điền ngày thật)


## 4. Môi trường

```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 5. Dữ liệu

Đặt file gốc `prosperLoanData.csv` (tải từ Kaggle — dán link thật vào đây)
vào `data/raw/prosperLoanData.csv`. Không sửa file này (giữ nguyên raw).

## 6. Chạy pipeline

```bash
python main.py
```

Kết quả sẽ được ghi ra:
- `data/processed/dataset_clean.csv` — dữ liệu đã làm sạch
- `outputs/tables/` — các bảng kết quả (missing report, cleaning trace,
  bảng kết quả 3 hypothesis)
- `outputs/figures/` — 3 biểu đồ chính (box plot, bar chart, scatter)

## 7. Chạy test

```bash
pytest -v tests/
```




