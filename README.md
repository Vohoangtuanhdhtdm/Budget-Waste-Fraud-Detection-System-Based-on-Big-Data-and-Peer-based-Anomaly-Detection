# 🏥 Health Insurance Fraud Detection (Big Data & Machine Learning)

![PySpark](https://img.shields.io/badge/Apache%20Spark-FDEE21?style=for-the-badge&logo=apachespark&logoColor=black)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-FF6F00?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Data Analysis](https://img.shields.io/badge/Data%20Analysis-150458?style=for-the-badge)

## 📌 Tổng quan dự án (Project Overview)
Gian lận bảo hiểm y tế là một vấn đề gây thất thoát hàng tỷ đô la mỗi năm đối với các tổ chức chính phủ và công ty bảo hiểm. Dự án này được thực hiện nhằm xử lý tập dữ liệu y tế khổng lồ (gần 10 triệu bản ghi) từ CMS (Centers for Medicare & Medicaid Services).

Thông qua việc kết hợp **Kiến thức Kinh tế Y tế** và sức mạnh của **Machine Learning trên PySpark**, hệ thống tự động tìm ra các bác sĩ có dấu hiệu kê khống giá dịch vụ, lạm dụng dịch vụ, hoặc có hành vi thu nhập bất thường so với khối lượng công việc thực tế. Hệ thống đặc biệt tập trung vào các chuyên khoa có giá trị kinh tế cao như Tim mạch (Cardiology) và Chẩn đoán hình ảnh (Diagnostic Radiology).

## 🛠 Tech Stack
- **Framework Xử lý Dữ liệu Lớn:** Apache Spark (PySpark SQL, DataFrames).
- **Machine Learning (PySpark MLlib):** Linear Regression, Principal Component Analysis (PCA), VectorAssembler, StandardScaler.
- **Trực quan hóa dữ liệu (Visualization):** Matplotlib, Seaborn, Pandas.

---

## 🔬 Phương pháp luận & Quy trình (Methodology)

Dự án được chia làm 2 giai đoạn cốt lõi với 3 mô hình đánh giá (Model A, B, C):

### Giai đoạn 1: Khám phá Dữ liệu & Phân tích Thiệt hại (EDA & Excess Spending)
- Lọc bỏ nhiễu, chuẩn hóa các Schema và xử lý các giá trị Null. Trích xuất hơn 1.5 triệu bản ghi của 2 chuyên khoa trọng điểm.

### Giai đoạn 2: Machine Learning Điều Tra Gian Lận
Sử dụng các kỹ thuật học máy để phác họa chân dung của một kẻ gian lận:

#### 1. Mô hình A (Excess Spending)
Tìm ra mức giá tiêu chuẩn của thị trường (**Peer Price**) theo từng mã dịch vụ tại từng bang. Sau đó, so sánh giá yêu cầu của từng bác sĩ (`submitted_price`) với mức giá chuẩn này. 
- **Công thức:** `Tiền lãng phí = (Giá bác sĩ đòi - Giá chuẩn thị trường) * Tổng số dịch vụ`.

#### 2. Mô hình B (Linear Regression - Tìm kiếm thu nhập bất thường)
- **Tư tưởng:** Doanh thu của một bác sĩ phải tỷ lệ thuận một cách tuyến tính với Khối lượng công việc (số lượng dịch vụ) và Lượng khách hàng (số lượng bệnh nhân).
- **Cách thức:** Dùng PySpark MLlib huấn luyện mô hình `Linear Regression` học quy luật của thị trường.
- `Doanh thu dự đoán = (A * số dịch vụ) + (B * số lượng bệnh nhân) + C`
- **Kết quả:** Điểm `Model B Score` được tính bằng cách lấy *Thu nhập thực tế* trừ đi *Doanh thu dự đoán*. Những người có điểm chênh lệch dương khổng lồ chính là đối tượng đang kiếm tiền bất minh.

#### 3. Mô hình C (Principal Component Analysis - Kẻ dị biệt)
- **Tư tưởng:** Kẻ gian lận sẽ luôn có cách lách luật, nhưng họ không thể giấu được việc phá vỡ cấu trúc phân phối tự nhiên của tập dữ liệu.
- **Cách thức:** Nhóm dữ liệu theo từng bác sĩ với 3 Vector cốt lõi: `[total_services, total_beneficiaries, total_actual_payment]`. Dùng `StandardScaler` chuẩn hóa dữ liệu về độ lệch chuẩn, sau đó đẩy qua mô hình **PCA (k=1)**.
- **Kết quả (`Model C Score`):** Biến đổi không gian đa chiều về 1 trục duy nhất. 
  - Điểm `0 - 3`: Bác sĩ bình thường (chiếm 99%).
  - Điểm `10 - 50+`: Các "kẻ dị biệt" (Anomalies) cần được thanh tra ngay lập tức.

---

## 📊 Kết quả đạt được (Results)
Dự án đã thành công trong việc khoanh vùng chính xác Top các bác sĩ gây thất thoát lớn nhất cho quỹ bảo hiểm y tế. 

- **Trực quan hóa:** Các đồ thị Scatter Plot (sử dụng Seaborn) phân loại rõ ràng mức độ dị biệt của từng bác sĩ dựa trên màu sắc (Hue) và kích thước (Size) của `Model C Score`. 
- Các bác sĩ dị biệt (nằm tách biệt hoàn toàn so với cụm phân phối chuẩn) đã được highlight tự động kèm theo số tiền thu nhập bất thường để phục vụ cho công tác thanh tra.

---

## 🚀 Cài đặt và Sử dụng (Installation & Usage)
Dự án được chạy trên môi trường **Google Colab** để tận dụng tài nguyên bộ nhớ cho PySpark.

1. Clone repository này về máy:
   ```bash
   git clone [https://github.com/your-username/health-insurance-fraud-detection.git](https://github.com/your-username/health-insurance-fraud-detection.git)
