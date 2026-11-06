# Day 01 — AI Foundation: Phân loại SMS Spam/Ham

Bài thực hành so sánh Bag-of-Words + Logistic Regression, MLP, MiniLM + đầu phân loại và LLM trên cùng bài toán lọc tin nhắn rác.

## Bài nộp

- [Báo cáo cá nhân](report/individual/2A202603688_TranMinhTan.md)
- [Notebook đã chạy, có bảng và biểu đồ](report/individual/day01_lab.ipynb)
- [Số liệu và kết quả kiểm tra](report/individual/lab1_report.json)
- [Ảnh tính tay](report/individual/hand_calc.jpg)

SEED: **1281**. Kết quả kiểm tra: **15/15 đạt**; có đủ **10 dự đoán** và thử thách kNN với k = 1, 5, 25.

LLM sử dụng Gemini 3.5 Flash-Lite và Gemini 3.1 Flash-Lite. Chi phí trong báo cáo là ước tính theo bảng giá tham khảo; độ trễ phụ thuộc runtime và điều kiện gọi API. Việc dùng AI được khai báo trong báo cáo và notebook.

## Chạy lại

Tải notebook vào Google Colab và chạy từ đầu. Phần LLM đọc key qua Colab Secret `GEMINI_API_KEY` với Notebook access được bật. Chạy lại thí nghiệm sẽ dùng quota API; đọc kết quả đã lưu không cần key.

## Nguồn

- [Repository đề bài](https://github.com/VinUni-AI20k/K5-L3-Day-01-AI-Foundation-Lab)
- [Nguồn dữ liệu và giấy phép bên thứ ba](THIRD_PARTY_NOTICES.md)
