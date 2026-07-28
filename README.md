# Medical AI Agent for Vietnamese Clinical NLP

Hệ thống AI xử lý ngôn ngữ tự nhiên (NLP) chuyên sâu trong lĩnh vực y khoa tiếng Việt. Dự án sử dụng mô hình pre-trained **PhoBERT (vinai/phobert-base)**, tối ưu hóa fine-tuning thông qua PyTorch và Hugging Face để nhận diện các thực thể y học (Named Entity Recognition - NER) gồm: **Triệu chứng (SYMPTOM)**, **Bệnh lý (DISEASE)**, **Thuốc (MEDICINE)**, và **Chỉ định cận lâm sàng/Xét nghiệm (TEST)**.

---

## 1. Kiến trúc hệ thống (Architecture)

Hệ thống được thiết kế theo cấu trúc modular, tách biệt rõ ràng giữa tiền xử lý dữ liệu, huấn luyện mô hình, đánh giá hiệu năng và triển khai API.

```text
Mediacal-AI-Agent/
├── data/                    # Chứa tập dữ liệu huấn luyện, kiểm thử (JSON format)
├── models/                  # Lưu trữ trọng số mô hình sau khi huấn luyện (Checkpoint)
├── logs/                    # Chứa thông tin ghi log xoay vòng (Rotating Logs)
├── src/
│   ├── api/                 # Cổng giao tiếp REST API (FastAPI)
│   ├── agent/               # Logic Agent nâng cao và tích hợp vector search
│   ├── training/            # Pipeline huấn luyện, định nghĩa Dataset và đánh giá (seqeval)
│   ├── inference/           # Logic xử lý inference và căn chỉnh offset thực thể gốc
│   └── utils/               # Trình tách từ (Word Segmenter), cấu hình (Config) và Logger
├── requirements.txt         # Thư viện phụ thuộc
├── main.py                  # Entrypoint điều phối hệ thống thông qua CLI
└── Dockerfile               # Cấu hình container hóa hỗ trợ GPU CUDA
```

### Luồng xử lý dữ liệu (Inference Pipeline Flow)
1. **Raw Text Input**: Người dùng gửi văn bản lâm sàng thô (ví dụ: *"Bệnh nhân sốt cao"*).
2. **Word Segmentation**: Sử dụng `underthesea` chuyển văn bản thô thành các từ ghép tiếng Việt ngăn cách bởi dấu gạch dưới (`"Bệnh_nhân sốt_cao"`).
3. **Subword Tokenization**: PhoBERT Tokenizer chuyển từ ghép thành subwords và lập bản đồ `word_ids`.
4. **GPU Forward Pass**: Mô hình `PhoBERT-NER` dự đoán nhãn BIO trên thiết bị CUDA (`RTX 4060`).
5. **BIO Post-processing & Span Alignment**: Căn chỉnh lại kết quả BIO sang offset ký tự trong chuỗi văn bản gốc để xuất ra vị trí `start` và `end` chính xác của thực thể.

---

## 2. Yêu cầu hệ thống & Cài đặt GPU CUDA

### Cấu hình phần cứng tối thiểu:
* **GPU**: NVIDIA GPU (Khuyên dùng RTX 3060/4060 trở lên)
* **RAM**: 16 GB trở lên
* **CUDA**: CUDA Toolkit 11.8 hoặc 12.x trở lên

### Hướng dẫn thiết lập môi trường Anaconda:

```bash
# 1. Tạo môi trường ảo với Python 3.10
conda create -n medical-nlp python=3.10 -y
conda activate medical-nlp

# 2. Cài đặt PyTorch với hỗ trợ CUDA (Ví dụ CUDA 11.8 hoặc 12.1 tương thích RTX 4060)
# Xem lệnh chi tiết tại: https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Cài đặt các thư viện phụ thuộc của dự án
pip install -r requirements.txt
```

Để kiểm tra thiết lập CUDA thành công:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0))"
```

---

## 3. Hướng dẫn chạy hệ thống (Execution Guide)

Hệ thống được vận hành toàn bộ thông qua file điều phối `main.py`.

### Bước 3.1: Tạo dữ liệu huấn luyện mẫu (Synthetic Data Generation)
Sinh tự động tập dữ liệu y khoa tiếng Việt chất lượng cao ở định dạng BIO để huấn luyện mô hình:
```bash
python main.py --mode generate-data
```
*Kết quả:* Tạo ra các file `train.json`, `val.json`, và `test.json` nằm trong thư mục `data/`.

### Bước 3.2: Huấn luyện mô hình (Model Fine-tuning)
Fine-tuning PhoBERT trên GPU CUDA:
```bash
python main.py --mode train
```
*Mô tả:* Script tự động cấu hình Mixed Precision Training (`fp16`), tự động chuyển mô hình lên CUDA, theo dõi hàm loss, tính điểm F1 sau mỗi epoch và lưu mô hình tốt nhất vào thư mục `models/phobert-medical-ner`.

### Bước 3.3: Đánh giá mô hình (Model Evaluation)
Đánh giá mô hình đã lưu trên tập dữ liệu kiểm thử độc lập:
```bash
python main.py --mode evaluate
```
*Kết quả:* In ra chi tiết bảng phân loại thực thể (`Classification Report` của thư viện `seqeval`) bao gồm Precision, Recall, F1-Score cho từng loại thực thể (`DISEASE`, `SYMPTOM`, `MEDICINE`, `TEST`).

### Bước 3.4: Dự đoán qua CLI (Command Line Inference)
Dự đoán nhanh thực thể y khoa từ văn bản thô bằng CLI:
```bash
python main.py --mode predict --text "Bệnh nhân bị viêm phổi có triệu chứng sốt cao và ho khan, bác sĩ kê Panadol uống."
```

---

## 4. Triển khai API Server (FastAPI API Usage)

Khởi động FastAPI server chạy trên cổng `8000`:
```bash
python main.py --mode api
```

### API Endpoint chi tiết:

#### 1. Kiểm tra trạng thái mô hình
* **Route**: `GET /health`
* **Response**:
  ```json
  {
    "status": "healthy",
    "device": "cuda",
    "model": "vinai/phobert-base"
  }
  ```

#### 2. Nhận diện thực thể y khoa (NER Inference)
* **Route**: `POST /predict`
* **Request Body**:
  ```json
  {
    "text": "Bệnh nhân sốt cao kèm đau họng nhẹ, chẩn đoán mắc sốt xuất huyết và uống Paracetamol."
  }
  ```
* **Response**:
  ```json
  {
    "text": "Bệnh nhân sốt cao kèm đau họng nhẹ, chẩn đoán mắc sốt xuất huyết và uống Paracetamol.",
    "entities": [
      {
        "text": "sốt cao",
        "type": "SYMPTOM",
        "score": 0.9854,
        "start": 10,
        "end": 17
      },
      {
        "text": "đau họng",
        "type": "SYMPTOM",
        "score": 0.9621,
        "start": 22,
        "end": 30
      },
      {
        "text": "sốt xuất huyết",
        "type": "DISEASE",
        "score": 0.9912,
        "start": 48,
        "end": 62
      },
      {
        "text": "Paracetamol",
        "type": "MEDICINE",
        "score": 0.9985,
        "start": 71,
        "end": 82
      }
    ],
    "latency_ms": 32.45
  }
  ```

---

## 5. Triển khai bằng Docker hỗ trợ GPU (Docker Deployment)

Để triển khai Docker tận dụng GPU NVIDIA của máy chủ vật lý, bạn cần cài đặt **NVIDIA Container Toolkit** trên hệ điều hành máy chủ.

### Dockerfile

```dockerfile
# Sử dụng base image hỗ trợ CUDA của NVIDIA
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Cài đặt Python 3.10 và pip
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Sao chép các tệp tin cấu hình
COPY requirements.txt .

# Cài đặt thư viện Python (Đặc biệt cấu hình PyTorch CUDA)
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn vào container
COPY src/ ./src
COPY main.py .
COPY data/ ./data
COPY models/ ./models

# Expose cổng của API
EXPOSE 8000

# Chạy FastAPI trên môi trường production
CMD ["python3", "main.py", "--mode", "api"]
```

### Docker Compose (`docker-compose.yml`)

```yaml
version: "3.8"
services:
  medical-nlp-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_NAME=vinai/phobert-base
      - API_PORT=8000
      - API_HOST=0.0.0.0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: always
```

Chạy lệnh sau để khởi động container:
```bash
docker-compose up --build -d
```
