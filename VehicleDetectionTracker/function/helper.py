import math
from .paddleocr_wrapper import create_paddleocr_reader

# license plate type classification helper function
def linear_equation(x1, y1, x2, y2):
    b = y1 - (y2 - y1) * x1 / (x2 - x1)
    a = (y1 - b) / x1
    return a, b

def check_point_linear(x, y, x1, y1, x2, y2):
    a, b = linear_equation(x1, y1, x2, y2)
    y_pred = a*x+b
    return(math.isclose(y_pred, y, abs_tol = 3))

# detect character and number in license plate using PaddleOCR
def read_plate(paddleocr_reader, im):
    """
    Đọc biển số xe sử dụng PaddleOCR thay vì YOLO
    
    Args:
        paddleocr_reader: PaddleOCR reader instance
        im: Input image
        
    Returns:
        str: Biển số xe được nhận dạng
    """
    try:
        # Sử dụng PaddleOCR để đọc text
        license_plate = paddleocr_reader.read_license_plate(im)
        
        # Nếu không đọc được, thử với các góc xoay khác nhau
        if license_plate == "unknown":
            import cv2
            import numpy as np
            
            # Thử xoay ảnh các góc khác nhau
            angles = [90, 180, 270]
            for angle in angles:
                # Xoay ảnh
                height, width = im.shape[:2]
                center = (width // 2, height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated_im = cv2.warpAffine(im, rotation_matrix, (width, height))
                
                # Thử đọc với ảnh đã xoay
                rotated_text = paddleocr_reader.read_license_plate(rotated_im)
                if rotated_text != "unknown":
                    license_plate = rotated_text
                    break
        
        return license_plate
        
    except Exception as e:
        print(f"ERROR: Error reading license plate with PaddleOCR: {e}")
        return "unknown"

# Legacy function để tương thích với code cũ (sử dụng YOLO)
def read_plate_yolo(yolo_license_plate, im):
    """
    Legacy function sử dụng YOLO (để tương thích với code cũ)
    """
    LP_type = "1"
    results = yolo_license_plate(im)
    bb_list = results.pandas().xyxy[0].values.tolist()
    if len(bb_list) == 0 or len(bb_list) < 7 or len(bb_list) > 10:
        return "unknown"
    center_list = []
    y_mean = 0
    y_sum = 0
    for bb in bb_list:
        x_c = (bb[0]+bb[2])/2
        y_c = (bb[1]+bb[3])/2
        y_sum += y_c
        center_list.append([x_c,y_c,bb[-1]])

    # find 2 point to draw line
    l_point = center_list[0]
    r_point = center_list[0]
    for cp in center_list:
        if cp[0] < l_point[0]:
            l_point = cp
        if cp[0] > r_point[0]:
            r_point = cp
    for ct in center_list:
        if l_point[0] != r_point[0]:
            if (check_point_linear(ct[0], ct[1], l_point[0], l_point[1], r_point[0], r_point[1]) == False):
                LP_type = "2"

    y_mean = int(int(y_sum) / len(bb_list))

    # 1 line plates and 2 line plates
    line_1 = []
    line_2 = []
    license_plate = ""
    if LP_type == "2":
        for c in center_list:
            if int(c[1]) > y_mean:
                line_2.append(c)
            else:
                line_1.append(c)
        for l1 in sorted(line_1, key = lambda x: x[0]):
            license_plate += str(l1[2])
        license_plate += "-"
        for l2 in sorted(line_2, key = lambda x: x[0]):
            license_plate += str(l2[2])
    else:
        for l in sorted(center_list, key = lambda x: x[0]):
            license_plate += str(l[2])
    return license_plate