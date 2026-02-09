from ultralytics import YOLO
import cv2

results = {}

# load models
# coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('model/license_plate_detector.pt')

vehicles = [2, 3, 5, 7]

image_path = r"D:\TrucNV\vehicle_detection_tracker\screenshots\20260206\164153_2\vehicle_frame_process_20260206_164153_916.png"
frame = cv2.imread(str(image_path))
# detect license plates
license_plates = license_plate_detector(frame)[0]
for license_plate in license_plates.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = license_plate
    # crop license plate
    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

    cv2.imshow("license_plate_crop", license_plate_crop)
    cv2.waitKey(0)
