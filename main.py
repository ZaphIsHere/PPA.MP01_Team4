from src import data_cleaner, analyze, visualizer
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def main() -> None:
    print("=" * 60)
    print("BƯỚC 1: LÀM SẠCH VÀ CHUẨN BỊ DỮ LIỆU")
    print("=" * 60)
    data_cleaner.main()
    
    print("\n" + "=" * 60)
    print("BƯỚC 2: PHÂN TÍCH VÀ KIỂM ĐỊNH GIẢ THUYẾT")
    print("=" * 60)
    analyze.main()
    
    print("\n" + "=" * 60)
    print("BƯỚC 3: TRỰC QUAN HÓA DỮ LIỆU")
    print("=" * 60)
    visualizer.main()
    
    print("\n[SUCCESS] Hoàn tất toàn bộ pipeline! Kết quả đã được lưu trong thư mục outputs và data/processed.")

if __name__ == "__main__":
    main()
