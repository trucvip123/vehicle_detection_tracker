# Hướng dẫn cấu hình

File `config.yaml` cho phép bạn dễ dàng điều chỉnh các tham số của hệ thống mà không cần sửa code.

## Cách sử dụng

1. Mở file `VehicleDetectionTracker/config.yaml`
2. Chỉnh sửa các giá trị theo nhu cầu
3. Lưu file và chạy lại ứng dụng

## Các nhóm cấu hình

### 1. Detection Settings (YOLO Vehicle Detection)

```yaml
detection:
  confidence: 0.4        # Ngưỡng confidence (0.0-1.0), giá trị thấp = phát hiện nhiều hơn
  iou: 0.45             # Ngưỡng IoU cho NMS, giá trị thấp = loại bỏ nhiều box trùng lặp
  image_size: 1280      # Kích thước ảnh xử lý (640, 1280, 1920)
  vehicle_classes: [2, 5, 6, 7, 8]  # Class IDs: car=2, bus=5, train=6, truck=7, boat=8
```

**Gợi ý điều chỉnh:**
- Giảm `confidence` (0.2-0.25) nếu muốn phát hiện nhiều phương tiện hơn
- Tăng `confidence` (0.4-0.5) nếu có quá nhiều false positives
- Tăng `image_size` (1920) để chính xác hơn nhưng chậm hơn
- Giảm `image_size` (640) để nhanh hơn nhưng ít chính xác hơn

### 2. Tracking Settings

```yaml
tracking:
  tracker_type: "bytetrack.yaml"  # Loại tracker: bytetrack, botsort, ocsort
  max_history_length: 30           # Số vị trí tối đa lưu trong lịch sử
```

### 3. License Plate Detection

```yaml
plate_detection:
  image_size: 640       # Kích thước ảnh xử lý cho plate detector
  min_width: 40         # Chiều rộng tối thiểu (pixels)
  min_height: 20        # Chiều cao tối thiểu (pixels)
```

**Gợi ý:**
- Giảm `min_width` nếu camera xa, biển số nhỏ
- Tăng `min_width` để loại bỏ các detection sai

### 4. OCR Settings

```yaml
ocr:
  confidence_threshold: 0.3  # Ngưỡng confidence cho OCR
  min_length: 6              # Độ dài tối thiểu (ký tự)
  max_length: 15             # Độ dài tối đa (ký tự)
```

### 5. RTSP Stream Settings

```yaml
rtsp:
  max_reconnect_attempts: 10      # Số lần thử kết nối lại (0 = vô hạn)
  reconnect_delay: 1              # Thời gian chờ giữa các lần thử (giây)
  open_timeout_ms: 5000           # Timeout mở stream (ms)
  read_timeout_ms: 5000           # Timeout đọc frame (ms)
  max_consecutive_failures: 10    # Số lỗi liên tiếp tối đa trước khi dừng
```

**Gợi ý:**
- Tăng `reconnect_delay` nếu mạng không ổn định
- Tăng `open_timeout_ms` nếu camera chậm khởi động

### 6. Telegram Bot Settings

```yaml
telegram:
  max_retries: 3              # Số lần thử lại tối đa
  request_timeout: 15         # Timeout cho mỗi request (giây)
  retry_delays: [2, 4, 8]     # Thời gian chờ giữa các lần thử (giây)
  dns_retry_delay: 2          # Thời gian chờ cho lỗi DNS (giây)
```

### 7. Threading Settings

```yaml
threading:
  max_workers: 8  # Số worker threads cho background tasks
```

**Gợi ý:**
- Tăng `max_workers` nếu CPU mạnh và có nhiều phương tiện đồng thời
- Giảm `max_workers` nếu CPU yếu hoặc muốn tiết kiệm tài nguyên

### 8. File Paths

```yaml
paths:
  yolo_model: "yolov8n.pt"                    # Model YOLO
  plate_model: "model/LP_detector.pt"         # Model plate detector
  screenshots_dir: "screenshots"              # Thư mục lưu ảnh
  excel_output: "vehicle_data.xlsx"          # File Excel output
```

### 9. Display Settings

```yaml
display:
  show_vehicle_boxes: false        # Hiển thị box quanh phương tiện
  show_plates_in_corner: true      # Hiển thị biển số ở góc màn hình
  stream_frame_size: null          # Resize frame (null = không resize)
```

**Ví dụ resize:**
```yaml
stream_frame_size: [1280, 720]  # Width x Height
```

### 10. Advanced Settings

```yaml
advanced:
  initialize_all_models: true              # Khởi tạo tất cả models ngay
  send_telegram_notification: true         # Gửi thông báo Telegram
  min_frames_for_plate_detection: 5        # Số frame tối thiểu trước khi detect plate
```

## Ví dụ điều chỉnh cho các tình huống

### Tình huống 1: Camera xa, phương tiện nhỏ

```yaml
detection:
  confidence: 0.25          # Giảm để phát hiện phương tiện nhỏ
  image_size: 1920          # Tăng để chính xác hơn

plate_detection:
  min_width: 30             # Giảm để nhận biển số nhỏ
```

### Tình huống 2: Mạng không ổn định

```yaml
rtsp:
  reconnect_delay: 3        # Tăng thời gian chờ
  max_consecutive_failures: 20  # Cho phép nhiều lỗi hơn

telegram:
  request_timeout: 30       # Tăng timeout
  retry_delays: [5, 10, 15] # Tăng thời gian giữa các lần thử
```

### Tình huống 3: Hiệu năng cao (GPU mạnh)

```yaml
detection:
  image_size: 1920          # Tăng độ phân giải
  confidence: 0.25          # Phát hiện nhiều hơn

threading:
  max_workers: 8            # Tăng số threads
```

### Tình huống 4: Tiết kiệm tài nguyên (CPU yếu)

```yaml
detection:
  image_size: 640           # Giảm độ phân giải
  confidence: 0.4           # Tăng ngưỡng để giảm số lượng detection

threading:
  max_workers: 2            # Giảm số threads
```

## Lưu ý

1. Sau khi chỉnh sửa config, cần restart ứng dụng để áp dụng thay đổi
2. Các giá trị mặc định đã được tối ưu cho hầu hết các trường hợp
3. Nếu config file không tồn tại, hệ thống sẽ sử dụng giá trị mặc định
4. Luôn backup config trước khi chỉnh sửa nhiều


