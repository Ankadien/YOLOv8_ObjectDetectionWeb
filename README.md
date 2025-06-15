# 🧠 Ứng dụng Web Phát Hiện Đối Tượng bằng YOLOv8

Đây là một ứng dụng web đơn giản và thân thiện cho việc phát hiện đối tượng trong ảnh, sử dụng mô hình **YOLOv8** kết hợp với **Streamlit** để xây dựng trang web.  
Người dùng có thể tải ảnh lên và xem kết quả phát hiện đối tượng với khung giới hạn và nhãn đối tượng.



## 🌟 Tính năng

- Tải ảnh lên (file tối đa 200MB, hỗ trợ file JPG, PNG, BMP, WEBP).
- Phát hiện đối tượng bằng mô hình YOLOv8 (`yolov8n.pt`).
- Giao diện web sử dụng bằng Streamlit.
- Có thể dễ dàng chạy tại máy tính cá nhân.



## 👤 Tác giả

**Nguyễn Hoài An**



## 📁 Cấu trúc thư mục

YOLOv8_ObjectDetectionWeb/
├── requirements.txt 
├── web.py 
├── yolov8n.pt 
└── README.md

yaml
Copy code



## 🚀 Hướng dẫn cách cài đặt và chạy ứng dụng webweb

### 1. Clone repository

```bash
git clone https://github.com/<ten-nguoi-dung>/YOLOv8_ObjectDetection_Web.git
cd YOLOv8_ObjectDetectionWeb
```

### 2. 📦 Cài đặt các thư viện cần thiết
```bash
Copy code
pip install -r requirements.txt
```

### 3. 🚀 Chạy ứng dụng web
```bash
Copy code
python -m streamlit run web.py
```


## 🎬 Video Demo
👉 Xem video demo cách hoạt động của ứng dụng tại đây:

