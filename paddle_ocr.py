#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#file paddleocr
"""
Module OCR sử dụng PaddleOCR - Tối ưu cho biển số Việt Nam
"""

import numpy as np
import cv2
import re
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

from config import (
    OCR_CORRECTIONS, PADDLEOCR_LANGUAGES, PADDLEOCR_USE_GPU,
    WARPED_PLATE_WIDTH, WARPED_PLATE_HEIGHT, WARPED_PLATE_HEIGHT_2_LINE
)


class PaddleOCRWrapper:
    """Wrapper cho PaddleOCR với tiền xử lý cho biển số Việt Nam"""
    
    def __init__(self):
        self.ocr = None
        self.available = False
        self._init_paddle_ocr()
    
    def _init_paddle_ocr(self):
        """Khởi tạo PaddleOCR với cấu hình tối ưu"""
        try:
            from paddleocr import PaddleOCR
            
            print("🚀 Đang khởi tạo PaddleOCR...")
            
            self.ocr = PaddleOCR(
                use_angle_cls=False,
                lang=PADDLEOCR_LANGUAGES,
                use_gpu=PADDLEOCR_USE_GPU,
                det_db_thresh=0.3,
                det_db_box_thresh=0.5,
                det_db_unclip_ratio=1.5,
                rec_batch_num=1,
                drop_score=0.5
            )
            self.available = True
            print("✅ PaddleOCR initialized successfully")
            print(f"   - Language: {PADDLEOCR_LANGUAGES}")
            print(f"   - GPU: {PADDLEOCR_USE_GPU}")
            
        except Exception as e:
            print(f"❌ Failed to initialize PaddleOCR: {e}")
            print("💡 Trying with minimal configuration...")
            try:
                from paddleocr import PaddleOCR
                self.ocr = PaddleOCR(use_angle_cls=False, lang='en')
                self.available = True
                print("✅ PaddleOCR initialized with minimal config")
            except Exception as e2:
                print(f"❌ Still failed: {e2}")
                self.available = False
    
    def _preprocess_for_ocr(self, plate_img: np.ndarray, is_two_lines: bool = False) -> np.ndarray:
        """Tiền xử lý ảnh chuyên cho PaddleOCR"""
        try:
            if len(plate_img.shape) == 3:
                gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = plate_img.copy()
            
            # CLAHE tăng độ tương phản
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Làm sắc nét
            kernel_sharpen = np.array([[-1, -1, -1],
                                        [-1, 9, -1],
                                        [-1, -1, -1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
            
            # Resize về kích thước chuẩn
            target_h = WARPED_PLATE_HEIGHT_2_LINE if is_two_lines else WARPED_PLATE_HEIGHT
            resized = cv2.resize(sharpened, (WARPED_PLATE_WIDTH, target_h), 
                               interpolation=cv2.INTER_CUBIC)
            
            # Chuyển về RGB
            rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            return rgb
            
        except Exception as e:
            print(f"⚠️ Preprocessing error: {e}")
            return plate_img
    
    def _correct_ocr_result(self, text: str, is_two_lines: bool = False) -> str:
        """Sửa lỗi OCR - GIỮ NGUYÊN DẤU GẠCH NGANG"""
        if not text:
            return text
        
        if '-' in text:
            parts = text.split('-')
            if len(parts) == 2:
                prefix = parts[0]
                suffix = parts[1]
                
                prefix_clean = re.sub(r'[^A-Za-z0-9]', '', prefix.upper())
                suffix_clean = re.sub(r'[^A-Za-z0-9.]', '', suffix.upper())
                
                # Sửa lỗi trong prefix
                corrected_prefix = []
                for i, char in enumerate(prefix_clean):
                    if i < 2:
                        if char in 'OQ': corrected_prefix.append('0')
                        elif char in 'IL': corrected_prefix.append('1')
                        elif char == 'Z': corrected_prefix.append('2')
                        elif char == 'S': corrected_prefix.append('5')
                        elif char == 'B': corrected_prefix.append('8')
                        elif char == 'G': corrected_prefix.append('6')
                        elif char == 'P': corrected_prefix.append('9')
                        elif char == 'R': corrected_prefix.append('4')
                        elif char.isdigit(): corrected_prefix.append(char)
                        else: corrected_prefix.append('0')
                    else:
                        corrected_prefix.append(char)
                
                # Sửa lỗi trong suffix
                corrected_suffix = []
                for char in suffix_clean:
                    if char in 'OQ': corrected_suffix.append('0')
                    elif char in 'IL': corrected_suffix.append('1')
                    elif char == 'Z': corrected_suffix.append('2')
                    elif char == 'S': corrected_suffix.append('5')
                    elif char == 'B': corrected_suffix.append('8')
                    elif char == 'G': corrected_suffix.append('6')
                    elif char == 'P': corrected_suffix.append('9')
                    elif char == 'R': corrected_suffix.append('4')
                    else: corrected_suffix.append(char)
                
                return f"{''.join(corrected_prefix)}-{''.join(corrected_suffix)}"
        
        clean_text = re.sub(r'[^A-Za-z0-9]', '', text.upper())
        return clean_text
    
    def _format_vietnam_plate(self, text: str, is_two_lines: bool = False) -> str:
        """Format biển số theo chuẩn Việt Nam"""
        if not text:
            return ""
        
        text = text.upper().strip()
        
        # Biển xe máy 2 dòng
        if is_two_lines:
            # Pattern 1: 29X5-077.77
            match = re.search(r'([0-9]{2})([A-Z])([0-9])-([0-9]{3})\.([0-9]{2})', text)
            if match:
                province, letter, num, num1, num2 = match.groups()
                return f"{province}{letter}{num}-{num1}.{num2}"
            
            # Pattern 2: 29X5-07777
            match2 = re.search(r'([0-9]{2})([A-Z])([0-9])-([0-9]{5})', text)
            if match2:
                province, letter, num, numbers = match2.groups()
                return f"{province}{letter}{num}-{numbers[:3]}.{numbers[3:]}"
            
            # Pattern 3: 29X507777
            match3 = re.search(r'([0-9]{2})([A-Z])([0-9])([0-9]{5})', text)
            if match3:
                province, letter, num, numbers = match3.groups()
                return f"{province}{letter}{num}-{numbers[:3]}.{numbers[3:]}"
        
        # Biển ô tô 1 dòng
        else:
            # Pattern 1: 51A-123.45
            match = re.search(r'([0-9]{2})([A-Z])-([0-9]{3})\.([0-9]{2})', text)
            if match:
                province, letter, num1, num2 = match.groups()
                return f"{province}{letter}-{num1}.{num2}"
            
            # Pattern 2: 51A-12345
            match2 = re.search(r'([0-9]{2})([A-Z])-([0-9]{5})', text)
            if match2:
                province, letter, numbers = match2.groups()
                return f"{province}{letter}-{numbers[:3]}.{numbers[3:]}"
            
            # Pattern 3: 51A12345
            match3 = re.search(r'([0-9]{2})([A-Z])([0-9]{5})', text)
            if match3:
                province, letter, numbers = match3.groups()
                return f"{province}{letter}-{numbers[:3]}.{numbers[3:]}"
        
        # Format dựa trên độ dài
        clean = re.sub(r'[^A-Z0-9]', '', text)
        
        if is_two_lines and len(clean) >= 8:
            return f"{clean[:3]}{clean[3]}-{clean[4:7]}.{clean[7:9]}"
        elif not is_two_lines and len(clean) >= 7:
            return f"{clean[:3]}-{clean[3:6]}.{clean[6:8]}"
        
        return text
    
    def recognize(self, plate_img: np.ndarray, is_two_lines: bool = False) -> Tuple[str, float]:
        """Nhận diện biển số với PaddleOCR"""
        if not self.available or self.ocr is None:
            print("⚠️ PaddleOCR not available")
            return "", 0.0
        
        try:
            processed_img = self._preprocess_for_ocr(plate_img, is_two_lines)
            cv2.imwrite("debug_processed.jpg", processed_img)
            
            result = self.ocr.ocr(processed_img)
            
            if not result or not result[0]:
                print("⚠️ PaddleOCR không trả về kết quả")
                return "", 0.0
            
            texts = []
            confidences = []
            
            for line in result:
                if line:
                    for detection in line:
                        if len(detection) >= 2:
                            text = detection[1][0]
                            confidence = detection[1][1]
                            texts.append(text)
                            confidences.append(confidence)
                            print(f"📝 OCR raw: '{text}' (conf={confidence:.2f})")
            
            if not texts:
                return "", 0.0
            
            # Xử lý ghép text
            if len(texts) == 2 and is_two_lines:
                combined_text = texts[0] + "-" + texts[1]
                print(f"📝 Combined (2 lines): '{combined_text}'")
            else:
                combined_text = ''.join(texts)
                print(f"📝 Combined: '{combined_text}'")
            
            avg_confidence = sum(confidences) / len(confidences)
            
            corrected = self._correct_ocr_result(combined_text, is_two_lines)
            print(f"📝 Corrected: '{corrected}'")
            
            formatted = self._format_vietnam_plate(corrected, is_two_lines)
            print(f"📝 Formatted: '{formatted}'")
            
            if formatted and len(formatted) >= 6:
                print(f"✅ Final: '{formatted}' (conf={avg_confidence:.2f})")
                return formatted, avg_confidence
            else:
                print(f"⚠️ Failed to format: '{formatted}'")
                return "", 0.0
                    
        except Exception as e:
            print(f"❌ PaddleOCR error: {e}")
            import traceback
            traceback.print_exc()
            return "", 0.0


# Singleton instance
_paddle_ocr = None


def get_paddle_ocr():
    """Lấy instance PaddleOCR (singleton)"""
    global _paddle_ocr
    if _paddle_ocr is None:
        _paddle_ocr = PaddleOCRWrapper()
    return _paddle_ocr