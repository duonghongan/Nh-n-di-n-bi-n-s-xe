#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module OCR sử dụng EasyOCR
"""

import cv2
import numpy as np
from typing import Tuple
from config import OCR_CONFIDENCE_THRESHOLD


def recognize_plate_easyocr(reader, plate_img: np.ndarray, is_two_lines: bool = False) -> Tuple[str, float]:
    """Nhận diện biển số với EasyOCR"""
    try:
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img.copy()
        
        # Tăng cường ảnh
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Resize về kích thước phù hợp
        h, w = enhanced.shape
        if is_two_lines:
            target_h = 140
        else:
            target_h = 80
        
        scale = target_h / h
        target_w = int(w * scale)
        resized = cv2.resize(enhanced, (target_w, target_h))
        
        # OCR
        result = reader.readtext(resized, 
                                 paragraph=False,
                                 width_ths=0.5,
                                 height_ths=0.5)
        
        if not result:
            return "", 0.0
        
        # Ghép kết quả
        texts = []
        confidences = []
        
        for detection in result:
            text = detection[1]
            conf = detection[2]
            
            if conf > OCR_CONFIDENCE_THRESHOLD:
                texts.append(text)
                confidences.append(conf)
        
        if not texts:
            return "", 0.0
        
        # Sắp xếp theo vị trí
        combined_text = ''.join(texts)
        avg_confidence = sum(confidences) / len(confidences)
        
        return combined_text, avg_confidence
        
    except Exception as e:
        print(f"Lỗi EasyOCR: {e}")
        return "", 0.0