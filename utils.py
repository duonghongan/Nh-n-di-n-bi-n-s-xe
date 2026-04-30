#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#file utils
"""
Module chứa các hàm tiện ích dùng chung
"""

import cv2
import numpy as np
import os
import re
from typing import Tuple, Optional
from config import MIN_PLATE_AREA, MAX_PLATE_AREA, MIN_EDGE_DENSITY, MAX_EDGE_DENSITY


def read_image(image_path: str) -> Optional[np.ndarray]:
    """Đọc ảnh từ file"""
    try:
        if not os.path.exists(image_path):
            print(f"File không tồn tại: {image_path}")
            return None
        
        image = cv2.imread(image_path)
        if image is None:
            from PIL import Image
            pil_image = Image.open(image_path)
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return image
    except Exception as e:
        print(f"Lỗi đọc ảnh {image_path}: {e}")
        return None


def preprocess_image_for_detection(image):
    """Tiền xử lý ảnh trước khi phát hiện biển số"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    return thresh


def check_plate_quality(plate_img: np.ndarray, edges_img: np.ndarray = None) -> Tuple[bool, str]:
    """Kiểm tra chất lượng ảnh biển số"""
    h, w = plate_img.shape[:2]
    area = h * w
    
    if area < MIN_PLATE_AREA:
        return False, f"Diện tích quá nhỏ: {area} < {MIN_PLATE_AREA}"
    
    if area > MAX_PLATE_AREA:
        return False, f"Diện tích quá lớn: {area} > {MAX_PLATE_AREA}"
    
    if edges_img is not None:
        edge_density = np.sum(edges_img > 0) / area
        if edge_density < MIN_EDGE_DENSITY:
            return False, f"Mật độ cạnh quá thấp: {edge_density:.3f}"
        if edge_density > MAX_EDGE_DENSITY:
            return False, f"Mật độ cạnh quá cao: {edge_density:.3f}"
    
    return True, "Chất lượng tốt"


def get_province_from_plate(plate_number: str) -> str:
    """Lấy tên tỉnh từ biển số"""
    from config import PROVINCE_MAPPING
    if not plate_number or len(plate_number) < 2:
        return "Không xác định"
    
    province_code = plate_number[:2]
    return PROVINCE_MAPPING.get(province_code, "Không xác định")


def normalize_plate_text(text: str) -> str:
    """Chuẩn hóa text biển số"""
    if not text:
        return ""
    
    text = text.upper()
    text = text.replace('I', '1').replace('L', '1')
    text = text.replace('O', '0').replace('Q', '0')
    text = text.replace('S', '5').replace('Z', '2')
    
    return re.sub(r'[^A-Z0-9-]', '', text)


def letterbox(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize ảnh giữ tỷ lệ"""
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h))
    
    result = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return result


def detect_and_filter_boxes(contours, min_area=1000, max_area=50000, aspect_ratio_range=(2, 6)):
    """Phát hiện và lọc các bounding box từ contours"""
    boxes = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        if aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
            boxes.append((x, y, w, h))
    
    return boxes


def sort_boxes_left_to_right(boxes):
    """Sắp xếp các box từ trái sang phải"""
    return sorted(boxes, key=lambda x: x[0])


def merge_overlapping_boxes(boxes, overlap_threshold=0.3):
    """Gộp các box chồng lấn nhau"""
    if not boxes:
        return []
    
    boxes = sorted(boxes, key=lambda x: x[0])
    merged = []
    
    for box in boxes:
        if not merged:
            merged.append(list(box))
        else:
            last = merged[-1]
            x_overlap = max(0, min(last[0] + last[2], box[0] + box[2]) - max(last[0], box[0]))
            area_overlap = x_overlap * min(last[3], box[3])
            area_last = last[2] * last[3]
            
            if area_overlap / area_last > overlap_threshold:
                last[0] = min(last[0], box[0])
                last[1] = min(last[1], box[1])
                last[2] = max(last[0] + last[2], box[0] + box[2]) - last[0]
                last[3] = max(last[1] + last[3], box[1] + box[3]) - last[1]
            else:
                merged.append(list(box))
    
    return [tuple(b) for b in merged]


def extract_text_from_boxes(image, boxes, config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'):
    """Trích xuất văn bản từ các box"""
    try:
        import pytesseract
    except ImportError:
        print("Chưa cài đặt pytesseract")
        return []
    
    texts = []
    for x, y, w, h in boxes:
        roi = image[y:y+h, x:x+w]
        if roi.size > 0:
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi
            
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(thresh, config=config)
            text = clean_text(text)
            texts.append(text)
    
    return texts


def clean_text(text):
    """Làm sạch văn bản nhận dạng được"""
    if not text:
        return ""
    return re.sub(r'[^A-Z0-9]', '', text.upper())


def validate_license_plate(text):
    """Kiểm tra tính hợp lệ của biển số"""
    if not text:
        return False
    
    patterns = [
        r'^[A-Z]{2}\d{3,5}$',
        r'^\d[A-Z]\d{4,5}$',
        r'^\d{2}[A-Z]\d{3,5}$'
    ]
    
    for pattern in patterns:
        if re.match(pattern, text):
            return True
    
    return len(text) >= 4


def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    """Vẽ các bounding box lên ảnh"""
    img_copy = image.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, thickness)
    return img_copy