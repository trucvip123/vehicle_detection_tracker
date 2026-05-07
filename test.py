from ultralytics import YOLO
import cv2

results = {}

# load models
# coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('model/license_plate_detector.pt')

vehicles = [2, 3, 5, 7]

image_path = r"screenshots\20260506\0628_13\vehicle_frame_20260506_062829_617.png"
frame = cv2.imread(str(image_path))

height, width = frame.shape[:2]
mid_height = height // 2

# Crop nửa dưới
bottom_half = frame[mid_height:, :, :]

# Lấy kích thước của bottom_half
bottom_height, bottom_width = bottom_half.shape[:2]

# Để thành hình vuông, dùng kích thước nhỏ hơn
square_size = min(bottom_height, bottom_width)

# Crop 2 bên để center
left = (bottom_width - square_size) // 2
right = left + square_size

square_frame = bottom_half[:, left:right, :]

cv2.imshow("license_plate_crop", square_frame)
cv2.waitKey(0)
