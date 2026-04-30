#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#file ocr
"""
Module OCR - Chỉ sử dụng PaddleOCR
"""

import cv2
import numpy as np
from typing import Tuple, Any
from paddle_ocr import get_paddle_ocr


def detect_two_lines(plate_img: np.ndarray) -> bool:
    """Phát hiện biển số 2 dòng dựa trên tỷ lệ"""
    h, w = plate_img.shape[:2]
    aspect_ratio = w / h
    
    if aspect_ratio < 1.5:
        return True
    elif aspect_ratio > 2.0:
        return False
    
    if len(plate_img.shape) == 3:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_img.copy()
    
    horizontal_projection = np.sum(gray, axis=1)
    threshold = np.mean(horizontal_projection) * 0.8
    
    middle_start = len(horizontal_projection) // 3
    middle_end = 2 * len(horizontal_projection) // 3
    middle_values = horizontal_projection[middle_start:middle_end]
    
    return np.min(middle_values) < threshold


def recognize_plate(reader: Any, plate_img: np.ndarray, 
                    processed_img: np.ndarray = None,
                    is_two_lines: bool = None,
                    use_corner_mask: bool = True) -> Tuple[str, float]:
    """
    Nhận diện biển số - Chỉ sử dụng PaddleOCR 
    """
    if is_two_lines is None:
        is_two_lines = detect_two_lines(plate_img)
    
    # Sử dụng hàm mới có hỗ trợ góc đỏ
    plate_text, confidence = recognize_plate_with_corners(plate_img, is_two_lines, use_corner_mask)
    
    # Nếu thất bại và có ảnh đã xử lý, thử lại
    if not plate_text and processed_img is not None:
        print("🔄 Thử lại với ảnh đã qua xử lý...")
        plate_text, confidence = recognize_plate_with_corners(processed_img, is_two_lines, use_corner_mask)
    
    return plate_text, confidence


# ==================== THÊM MỚI: NHẬN DIỆN ====================

def recognize_plate_with_corners(plate_img: np.ndarray, 
                                  is_two_lines: bool = False,
                                  use_corner_mask: bool = True) -> Tuple[str, float]:
    """
    Nhận diện biển số 
    
    Args:
        plate_img: Ảnh biển số đã crop
        is_two_lines: Có phải biển 2 dòng không
        use_corner_mask: Có sử dụng phát hiện góc đỏ không
        
    Returns:
        plate_text: Biển số nhận diện được
        confidence: Độ tin cậy
    """
    if plate_img is None or plate_img.size == 0:
        return "", 0.0
    
    paddle_ocr = get_paddle_ocr()
    
    if not paddle_ocr.available:
        print("❌ PaddleOCR không khả dụng")
        return "", 0.0
    
    # Phương thức 1: OCR trên ảnh gốc
    text1, conf1 = paddle_ocr.recognize(plate_img, is_two_lines)
    
    best_text = text1
    best_conf = conf1
    
    # Phương thức 2: OCR với phát hiện góc đỏ và tô màu ngoài biển số
    if use_corner_mask and (not best_text or best_conf < 0.7):
        print("🔴 Thử phương thức phát hiện góc đỏ...")
        
        try:
            from preprocessing import enhance_plate_with_mask
            
            # Tăng cường ảnh với mask
            enhanced_img, corners, mask = enhance_plate_with_mask(plate_img)
            
            if corners and len(corners) >= 4:
                print(f"   ✅ Phát hiện {len(corners)} góc đỏ")
                
                # OCR trên ảnh đã tô màu
                text2, conf2 = paddle_ocr.recognize(enhanced_img, is_two_lines)
                
                if text2 and conf2 > best_conf:
                    best_text = text2
                    best_conf = conf2
                    print(f"   ✅ Phương thức góc đỏ tốt hơn: '{text2}' (conf={conf2:.2f})")
                else:
                    print(f"   ⚠️ Phương thức góc đỏ: '{text2}' (conf={conf2:.2f})")
            else:
                print(f"   ⚠️ Không phát hiện đủ 4 góc đỏ")
                
        except Exception as e:
            print(f"   ⚠️ Lỗi khi xử lý góc đỏ: {e}")
    
    # Phương thức 3: Nếu vẫn thất bại, thử với chế độ ngược lại
    if not best_text:
        print("🔄 Thử lại với chế độ ngược lại...")
        text3, conf3 = paddle_ocr.recognize(plate_img, not is_two_lines)
        if text3 and conf3 > best_conf:
            best_text = text3
            best_conf = conf3
    
    return best_text, best_conf


def recognize_plate_multi_method(plate_img: np.ndarray, 
                                   is_two_lines: bool = False) -> dict:
    """
    Nhận diện với nhiều phương thức và trả về kết quả chi tiết
    
    Returns:
        Dictionary chứa kết quả từ các phương thức
    """
    result = {
        'original': {'text': '', 'confidence': 0.0},
        'corner_mask': {'text': '', 'confidence': 0.0, 'corners': []},
        'best': {'text': '', 'confidence': 0.0, 'method': ''}
    }
    
    paddle_ocr = get_paddle_ocr()
    
    if not paddle_ocr.available:
        return result
    
    # Phương thức 1: Ảnh gốc
    text1, conf1 = paddle_ocr.recognize(plate_img, is_two_lines)
    result['original']['text'] = text1
    result['original']['confidence'] = conf1
    
    best_text = text1
    best_conf = conf1
    best_method = 'original'
    
    # Phương thức 2: Phát hiện góc đỏ và tô màu
    try:
        from preprocessing import enhance_plate_with_mask
        
        enhanced_img, corners, mask = enhance_plate_with_mask(plate_img)
        
        if corners and len(corners) >= 4:
            text2, conf2 = paddle_ocr.recognize(enhanced_img, is_two_lines)
            result['corner_mask']['text'] = text2
            result['corner_mask']['confidence'] = conf2
            result['corner_mask']['corners'] = corners
            
            if text2 and conf2 > best_conf:
                best_text = text2
                best_conf = conf2
                best_method = 'corner_mask'
    except Exception as e:
        print(f"⚠️ Lỗi phương thức góc đỏ: {e}")
    
    # Kết quả tốt nhất
    result['best']['text'] = best_text
    result['best']['confidence'] = best_conf
    result['best']['method'] = best_method
    
    return result