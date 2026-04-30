#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#file preprocessing
"""
Module tiền xử lý ảnh cho nhận diện biển số xe
ĐÃ CẢI TIẾN: Phát hiện góc bằng minAreaRect + HoughLines + deskew dự phòng
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any, List


def crop_image_border(image, percent=2):
    """Cắt viền ảnh theo tỷ lệ phần trăm"""
    if image is None or image.size == 0:
        return image
    
    h, w = image.shape[:2]
    crop_h = int(h * percent / 100)
    crop_w = int(w * percent / 100)
    
    if crop_h >= h // 2:
        crop_h = h // 4
    if crop_w >= w // 2:
        crop_w = w // 4
    
    cropped = image[crop_h:h-crop_h, crop_w:w-crop_w]
    return cropped


def enhance_contrast(image):
    """Tăng cường độ tương phản cho ảnh"""
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(image)


def order_points(pts):
    """
    Sắp xếp 4 điểm theo thứ tự: top-left, top-right, bottom-right, bottom-left
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # Tính tổng và hiệu tọa độ
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    # Top-left: tổng nhỏ nhất
    rect[0] = pts[np.argmin(s)]
    # Bottom-right: tổng lớn nhất
    rect[2] = pts[np.argmax(s)]
    # Top-right: hiệu nhỏ nhất
    rect[1] = pts[np.argmin(diff)]
    # Bottom-left: hiệu lớn nhất
    rect[3] = pts[np.argmax(diff)]
    
    return rect


def perspective_correction(image, pts, target_width=None, target_height=None):
    """
    Thực hiện perspective correction (dựng ảnh) dựa trên 4 điểm góc
    
    Args:
        image: Ảnh đầu vào
        pts: 4 điểm góc (đã sắp xếp) [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        target_width: Chiều rộng đích (tự động nếu None)
        target_height: Chiều cao đích (tự động nếu None)
    
    Returns:
        warped: Ảnh đã được dựng thẳng
    """
    if pts is None or len(pts) != 4:
        return image
    
    # Tính kích thước đích dựa trên khoảng cách giữa các điểm
    if target_width is None:
        # Tính chiều rộng trung bình (trên + dưới)
        width_top = np.sqrt(((pts[0][0] - pts[1][0]) ** 2) + ((pts[0][1] - pts[1][1]) ** 2))
        width_bottom = np.sqrt(((pts[3][0] - pts[2][0]) ** 2) + ((pts[3][1] - pts[2][1]) ** 2))
        target_width = max(int(width_top), int(width_bottom))
    
    if target_height is None:
        # Tính chiều cao trung bình (trái + phải)
        height_left = np.sqrt(((pts[0][0] - pts[3][0]) ** 2) + ((pts[0][1] - pts[3][1]) ** 2))
        height_right = np.sqrt(((pts[1][0] - pts[2][0]) ** 2) + ((pts[1][1] - pts[2][1]) ** 2))
        target_height = max(int(height_left), int(height_right))
    
    # Giới hạn kích thước
    target_width = min(target_width, 500)
    target_height = min(target_height, 200)
    
    # Tạo 4 điểm đích (hình chữ nhật vuông góc)
    dst = np.array([
        [0, 0],
        [target_width - 1, 0],
        [target_width - 1, target_height - 1],
        [0, target_height - 1]
    ], dtype=np.float32)
    
    # Tính ma trận chuyển đổi perspective
    src = np.array(pts, dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    
    # Thực hiện warp
    warped = cv2.warpPerspective(image, M, (target_width, target_height))
    
    return warped


# ==================== CẢI TIẾN: TIỀN XỬ LÝ ĐỂ TÌM CONTOUR TỐT HƠN ====================
def preprocess_for_contour_detection(gray_image):
    """
    Tiền xử lý ảnh để tìm contour biển số: CLAHE + Sobel + Morphology
    """
    # CLAHE tăng cường độ tương phản
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_image)

    # Sobel theo hai hướng để phát hiện cạnh
    sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobelx, sobely)
    sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))

    # Canny cũng là một lựa chọn
    edges = cv2.Canny(enhanced, 50, 150)

    # Kết hợp cả hai
    combined_edges = cv2.bitwise_or(sobel_combined, edges)

    # Đóng các khoảng trống và giãn nở để liên kết cạnh
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated = cv2.dilate(closed, kernel, iterations=2)
    return dilated


def find_plate_contour(image, debug=False):
    """
    Tìm contour biển số cải tiến, sử dụng minAreaRect và tỷ lệ khung hình linh hoạt
    Trả về 4 điểm góc đã sắp xếp hoặc None
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    processed = preprocess_for_contour_detection(gray)
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_rect = None
    max_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:
            continue

        # Sử dụng minAreaRect để ước lượng hình chữ nhật bao quanh tốt nhất
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        w, h = rect[1]
        if w == 0 or h == 0:
            continue
        aspect_ratio = max(w, h) / min(w, h)
        # Tỷ lệ khung hình của biển số thường từ 2:1 đến 6:1 (có thể nới lỏng)
        if 2.0 < aspect_ratio < 6.0:
            if area > max_area:
                max_area = area
                best_rect = rect

    if best_rect is not None:
        pts = cv2.boxPoints(best_rect)
        pts = order_points(pts)
        if debug:
            print(f"✅ Tìm thấy 4 góc bằng minAreaRect: {pts}")
        return pts
    return None


def auto_perspective_correction(plate_img, debug=False):
    """
    Tự động phát hiện 4 góc và dựng ảnh biển số, có cơ chế dự phòng
    """
    # Phương pháp chính: tìm 4 góc bằng contour cải tiến
    corners = find_plate_contour(plate_img, debug)
    
    if corners is not None:
        warped = perspective_correction(plate_img, corners)
        if debug:
            h, w = plate_img.shape[:2]
            print(f"✅ Perspective correction thành công: {w}x{h}")
        return warped, True
    
    # Dự phòng: xoay ảnh đơn giản (deskew) nếu không tìm thấy góc
    if debug:
        print("⚠️ Không tìm thấy 4 góc, thực hiện xoay ảnh (deskew)...")
    deskewed = deskew_image(plate_img)
    return deskewed, False


def deskew_image(image):
    """
    Chỉnh góc nghiêng cho ảnh (xoay đơn giản) - Dùng khi không tìm thấy 4 góc
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Tìm góc nghiêng bằng phương pháp Hough Line
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    
    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            if -45 < angle < 45:
                angles.append(angle)
        
        if angles:
            median_angle = np.median(angles)
            if abs(median_angle) > 0.5:
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                rotated = cv2.warpAffine(image, M, (w, h),
                                         flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_REPLICATE)
                return rotated
    
    return image


def preprocess_plate_image(plate_img, do_perspective=True, do_enhance=True):
    """
    Tiền xử lý ảnh biển số - CÓ DỰNG ẢNH (perspective correction) cải tiến
    """
    result = {
        'original': plate_img,
        'gray': None,
        'processed': None,
        'warped': None,
        'binary': None,
        'edges': None
    }
    
    if plate_img is None or plate_img.size == 0:
        return result
    
    # Chuyển sang grayscale nếu cần
    if len(plate_img.shape) == 3:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_img.copy()
    
    result['gray'] = gray
    
    # Resize nếu ảnh quá nhỏ
    h, w = gray.shape
    if w < 200:
        scale = 400 / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        print(f"📐 Resize ảnh: {w}x{h} -> {new_w}x{new_h}")
    
    # Khử nhiễu
    denoised = cv2.medianBlur(gray, 3)
    
    # ===== QUAN TRỌNG: DỰNG ẢNH (PERSPECTIVE CORRECTION) =====
    if do_perspective:
        # Thử dựng ảnh bằng perspective correction cải tiến
        warped_img, success = auto_perspective_correction(denoised, debug=False)
        
        if success:
            result['warped'] = cv2.cvtColor(warped_img, cv2.COLOR_GRAY2BGR)
            # Dùng ảnh đã dựng cho các bước tiếp theo
            processed_img = warped_img
            print("✅ Đã dựng ảnh thành công (perspective correction)")
        else:
            # Fallback: xoay đơn giản
            rotated = deskew_image(denoised)
            result['warped'] = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)
            processed_img = rotated
            print("⚠️ Không tìm thấy 4 góc, dùng xoay đơn giản")
    else:
        result['warped'] = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        processed_img = denoised
    
    # Tăng cường độ tương phản
    if do_enhance:
        enhanced = enhance_contrast(processed_img)
        result['processed'] = enhanced
    else:
        result['processed'] = processed_img
    
    # Tạo ảnh nhị phân (kết hợp nhiều phương pháp)
    binary1 = cv2.adaptiveThreshold(result['processed'], 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)
    
    _, binary2 = cv2.threshold(result['processed'], 0, 255, 
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    binary = cv2.bitwise_or(binary1, binary2)
    
    # Làm sạch ảnh nhị phân
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    result['binary'] = binary
    
    # Phát hiện cạnh
    edges = cv2.Canny(result['processed'], 50, 150)
    result['edges'] = edges
    
    return result