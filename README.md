# 🚗 Car Purchase Decision Analysis: Random Forest & XAI (SHAP)

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange?style=for-the-badge&logo=scikitlearn)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployment-red?style=for-the-badge&logo=streamlit)
![XAI](https://img.shields.io/badge/XAI-SHAP_Interpretability-blueviolet?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

> **"From Statistical Theory to Practical AI Tools: Deciphering Market Demand through Explainable Machine Learning."**

## 📖 Introduction (Giới thiệu)

Dự án tập trung phân tích các yếu tố ảnh hưởng đến **Sức mua thị trường (Total)** trong ngành ô tô bằng phương pháp Machine Learning kết hợp Explainable AI (XAI). 

Thay vì dự báo Giá (một bài toán tương quan thuận hiển nhiên), dự án tập trung vào biến **Total** (Tổng lượng người mua), được sử dụng như một **Biến đại diện (Proxy)** cho sự chấp nhận của thị trường. 

Mục tiêu cuối cùng là xây dựng một công cụ có khả năng giải thích: *"Tại sao một cấu hình xe nhất định lại thu hút khách hàng?"* thông qua lăng kính của **Explainable AI (XAI)**.

## 🤖 Development Philosophy (Triết lý phát triển)

Dự án này được thực hiện theo quy trình AI-Assisted Development (Phát triển với sự hỗ trợ của AI).

Ưu tiên tính logic hệ thống và khả năng ứng dụng thực tế thay vì tối ưu hóa các chỉ số hiệu suất ảo, với các nguyên tắc:
- Triển khai theo quy trình: Load → Clean → Train → Tune → Evaluate → Explain → Deploy.
- Kiểm soát rò rỉ dữ liệu (Data Leakage): Loại bỏ các "đường tắt" của mô hình để buộc AI phải học các đặc trưng kỹ thuật thực sự.
- Sử dụng Baseline Model để định lượng bản chất của dữ liệu trước khi áp dụng các giải pháp phức tạp hơn. 
- Ưu tiên tính giải thích (Interpretability) thay vì chỉ tối đa hóa độ chính xác.

## 💡 Key Features & Problem Solving (Điểm nhấn Kỹ thuật)

### 1. 🛡️ Data Leakage Control (Chống rò rỉ dữ liệu - Khái niệm sống còn)
* **Vấn đề (The Pain Point):** Dữ liệu gốc có công thức $Total = Male + Female + Unknown$. Nếu giữ lại các biến thành phần này, mô hình sẽ thực hiện "phép cộng" thay vì "học đặc trưng", dẫn đến kết quả 100% vô giá trị trong thực tế.
* **Giải pháp:** Triệt tiêu hoàn toàn các biến rò rỉ và định danh. Buộc mô hình Random Forest phải thực sự suy luận từ các thông số kỹ thuật cốt lõi: **Price, Power, Engine CC và Manufacturer.**.

### 2. 📉 Baseline Benchmarking (Chứng minh bản chất phi tuyến)
* **Chiến lược:** Sử dụng **Linear Regression** làm mốc so sánh (Baseline).
* **Mục đích:** Kiểm chứng mức độ phi tuyến của dữ liệu và chứng minh dữ liệu này phức tạp và phi tuyến đến mức nào. 

### 3. 🔍 Explainable AI với SHAP (Giải mã hộp đen)
* Sử dụng **SHAP Values** để bóc tách mức độ đóng góp của từng đặc trưng:
    * **Push factors:** Các yếu tố thúc đẩy nhu cầu.
    * **Drag factors:** Các yếu tố kìm hãm quyết định khách hàng.

### 4. ☁️ Product-Oriented Deployment
* Hệ thống được thiết kế để triển khai tức thì trên **Streamlit Cloud**.
* Tích hợp cơ chế **Automated Artifact Retrieval**: Load trực tiếp artifact từ repository để đảm bảo đồng bộ môi trường, loại bỏ mọi rào cản về cấu hình thủ công.

## 📊 Results & Insights (Kết quả & Nhận định)

### 1. Model Performance
* **R-squared ($R^2$):** Đạt xấp xỉ **$75\%$** trên tập kiểm thử sau khi đã chặn đứng mọi rủi ro rò rỉ dữ liệu. Là một con số thực tế, phản ánh khả năng tổng quát hóa hợp lý trong bối cảnh dữ liệu nghiên cứu.

### 2. Market Insights (Phân tích SHAP)
* **Yếu tố tích cực:** Thương hiệu (**Ford**) và cấu hình **Hộp số (Transmission)** có tác động tích cực nhất đến sức mua dự kiến.
* **Yếu tố tác động ngược:** Trong phần lớn các phân khúc thực nghiệm, sự gia tăng quá mức của **Price, Power** và **Engine CC** có xu hướng làm giảm điểm tiềm năng thị trường (Total), phản ánh xu hướng ưu tiên tính kinh tế và hiệu quả sử dụng.

## 🚧 Domain Constraints & Future Roadmap (Phạm vi & Định hướng)

Dự án này được thiết kế tối ưu cho thị trường xe **Động cơ đốt trong (ICE)**, dựa trên bối cảnh dữ liệu giai đoạn 2022 - nơi các biến số kỹ thuật truyền thống như `Engine CC` (Dung tích xi-lanh) và `Transmission` (Hộp số) đóng vai trò định giá cốt lõi.

Tôi nhận thức rõ sự chuyển dịch sang **Xe điện (EVs)** và xác định đây là giới hạn hiện tại của mô hình:
* **Limitation:** Mô hình hiện tại sẽ không tối ưu cho xe điện do thiếu các đặc trưng chuyên biệt (Pin, Phạm vi hoạt động).

## 🚀 How to Run (Hướng dẫn chạy & Trải nghiệm)

Dự án được thiết kế với tính linh hoạt cao, cho phép người dùng lựa chọn giữa việc trải nghiệm nhanh sản phẩm cuối hoặc đi sâu vào mã nguồn nghiên cứu:

### 1. Live Interface (Trải nghiệm người dùng cuối)
Truy cập Dashboard tương tác thời gian thực đã được triển khai trên nền tảng Cloud:
👉 **[Interactive Demand Simulator (Streamlit Cloud)](https://car-purchase-decision-analysis-xai-bbmz6ofo9vegh6agsnvmzy.streamlit.app/)**

### 2. Developer Mode (Dành cho nhà phát triển)
Hệ thống mã nguồn được chia thành 3 module độc lập, có thể chạy trực tiếp trên **Google Colab** mà không cần cài đặt môi trường:

* **[Phase 1: Tiền xử lý & Khám phá dữ liệu]**:
  <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/higo-ai/Car-Purchase-Decision-Analysis-XAI/blob/main/01_Car_Price_Preprocessing_EDA.ipynb)
  *(Xem quy trình làm sạch dữ liệu, xử lý Data Leakage và phân tích EDA)*

* **[Phase 2: Huấn luyện & Giải thích XAI]**:
  <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/higo-ai/Car-Purchase-Decision-Analysis-XAI/blob/main/02_Car_Price_Training_SHAP.ipynb)
  *(Xem quá trình tối ưu hóa Random Forest và giải mã mô hình bằng SHAP)*

* **[Phase 3: Mô phỏng triển khai App]**:
  <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/higo-ai/Car-Purchase-Decision-Analysis-XAI/blob/main/03_Car_Price_App_Streamlit.ipynb)
  *(Thử nghiệm cơ chế deploy Web App thông qua ngrok tunnel)*

## 📝 Các bước thực hiện trên Colab:

1. Sau khi nhấn nút **Open in Colab**, chọn menu **Runtime -> Run all** (hoặc nhấn `Ctrl + F9`).
2. Hệ thống sẽ tự động kích hoạt quy trình **Automated Artifact Retrieval** (tự động tải Model và Data từ GitHub).
3. Đối với **Phase 3**, bạn cần nhập mã `Ngrok Authtoken` (miễn phí) để khởi tạo đường dẫn truy cập bảo mật.

## 📢 Acknowledgements & Data Source (Nguồn dữ liệu & Lời cảm ơn)

Nghiên cứu này sử dụng bộ dữ liệu **Car Buyers** từ Kaggle. Xin gửi lời cảm ơn chân thành đến tác giả vì đã chia sẻ nguồn dữ liệu quý giá này cho cộng đồng nghiên cứu khoa học dữ liệu.

* **Dataset:** [Car Buyers Dataset by M. Chaudhuri (2022)](https://www.kaggle.com/datasets/brijlaldhankour/car-buyers/data)
* **Original Source:** Kaggle
* **Core Technology:** Scikit-learn, SHAP, Streamlit, Pandas.

---

### 👤 Author
* **Developer:** Bui Tien Phat (Higo)
* **Contact:** higo.individual@gmail.com
* **Role:** AI Engineer / Data Scientist

---

## 📂 Project Structure (Cấu trúc thư mục)

```text
Car-Purchase-Decision-Analysis-XAI/
├── data/
│   ├── CarBuyers.csv                     # Dữ liệu thô ban đầu (Raw Dataset)
│   └── processed_carbuyers.csv           # Dữ liệu sạch (Anti-leakage) dùng cho huấn luyện
├── models/
│   ├── car_purchase_model.joblib         # Random Forest Model đã tối ưu
│   └── model_columns.joblib              # Cấu trúc vector đặc trưng
├── 01_Car_Price_Preprocessing_EDA.ipynb  # Phase 1: Tiền xử lý & Khám phá
├── 02_Car_Price_Training_SHAP.ipynb      # Phase 2: Huấn luyện & Giải thích XAI
├── 03_Car_Price_App_Streamlit.ipynb      # Phase 3: Mô phỏng triển khai (Colab)
├── app.py                                # Mã nguồn ứng dụng Streamlit Cloud
├── requirements.txt                      # Dependencies của hệ thống
├── LICENSE                               # Giấy phép bản quyền dự án (MIT License)
└── README.md                             # Tài liệu dự án
