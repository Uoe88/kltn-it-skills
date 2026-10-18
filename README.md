# Phân tích Kỹ năng IT từ Tin tuyển dụng

## Mô tả đề tài
Dự án phân tích dữ liệu tuyển dụng ngành IT nhằm xác định cấu trúc kỹ năng, xu hướng công nghệ và nhu cầu thị trường.

## Các câu hỏi nghiên cứu (RQ)
- **RQ1:** Phân tích tỷ lệ phủ và các tổ hợp kĩ năng của 5 ngành cốt lõi
- **RQ2:** Phân tích mật độ của kĩ năng AI/LLM, Cloud/DevOps, Ngoại ngữ (Anh, Pháp, Nhật)
- **RQ3:** Phân tích độ rộng và phức tạp của kĩ năng của 5 ngành cốt lõi.
- **RQ4:** Phân tích thành phố tập trung nhân sự bậc cao.

## Sơ đồ Pipeline
clean_jobs.py → build_skill_taxonomy.py → run_all.py → app.py


## Hướng dẫn sử dụng

### 1. Cài đặt môi trường
```bash
pip install -r requirements.txt