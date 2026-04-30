#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#file postprocessing
"""
Module chứa các hàm hậu xử lý và chuẩn hóa biển số Việt Nam
"""

import numpy as np
import cv2
import re
from typing import Tuple
from config import PROVINCE_MAPPING


def correct_ocr_errors(text: str) -> str:
    """Sửa lỗi OCR - GIỮ NGUYÊN KẾT QUẢ, chỉ loại bỏ ký tự đặc biệt"""
    if not text:
        return text
    
    result = re.sub(r'[^A-Za-z0-9-.]', '', text)
    return result.upper()


def clean_plate_numbers(text: str) -> str:
    """Làm sạch biển số - GIỮ NGUYÊN ĐỊNH DẠNG, chỉ chuẩn hóa phần số"""
    if not text:
        return text
    
    text = re.sub(r'[^A-Za-z0-9-.]', '', text.upper())
    
    if '-' in text and '.' in text:
        parts = text.split('-')
        if len(parts) == 2:
            prefix = parts[0]
            suffix = parts[1]
            if '.' in suffix:
                suffix_parts = suffix.split('.')
                if len(suffix_parts) == 2:
                    num1 = re.sub(r'[^0-9]', '', suffix_parts[0])
                    num2 = re.sub(r'[^0-9]', '', suffix_parts[1])
                    if len(num1) == 3 and len(num2) == 2:
                        return f"{prefix}-{num1}.{num2}"
    
    if '-' in text:
        parts = text.split('-')
        if len(parts) == 2:
            prefix = parts[0]
            suffix = re.sub(r'[^0-9]', '', parts[1])
            if len(suffix) == 5:
                return f"{prefix}-{suffix[:3]}.{suffix[3:]}"
            elif len(suffix) == 4:
                return f"{prefix}-{suffix}"
            elif len(suffix) == 3:
                return f"{prefix}-{suffix}0"
    
    match = re.search(r'([A-Z0-9]+)([0-9]{4,5})', text)
    if match:
        prefix = match.group(1)
        numbers = match.group(2)
        if len(numbers) == 5:
            return f"{prefix}-{numbers[:3]}.{numbers[3:]}"
        elif len(numbers) == 4:
            return f"{prefix}-{numbers}"
    
    return text


def format_plate_number(plate_number: str, is_two_lines: bool = False) -> str:
    """Định dạng lại biển số theo chuẩn Việt Nam"""
    if not plate_number:
        return ""
    
    plate = plate_number.upper().strip()
    
    if '-' in plate and '.' in plate:
        parts = plate.split('-')
        if len(parts) == 2:
            prefix = parts[0]
            suffix = parts[1]
            if len(prefix) >= 3 and prefix[:2].isdigit():
                if '.' in suffix and len(suffix.replace('.', '')) == 5:
                    return plate
    
    clean = re.sub(r'[^A-Z0-9]', '', plate)
    
    if is_two_lines and len(clean) >= 8:
        return f"{clean[:3]}{clean[3]}-{clean[4:7]}.{clean[7:9]}"
    elif not is_two_lines and len(clean) >= 7:
        return f"{clean[:3]}-{clean[3:6]}.{clean[6:8]}"
    
    return clean


def validate_plate_format(plate_number: str) -> Tuple[bool, str]:
    """Kiểm tra biển số có đúng định dạng Việt Nam không"""
    if not plate_number:
        return False, "Biển số rỗng"
    
    if '-' not in plate_number:
        return False, "Thiếu dấu gạch ngang"
    
    parts = plate_number.split('-')
    if len(parts) != 2:
        return False, "Sai định dạng (cần 1 dấu -)"
    
    prefix, suffix = parts
    suffix_clean = suffix.replace('.', '')
    
    if len(prefix) < 2:
        return False, "Prefix quá ngắn"
    
    if not prefix[:2].isdigit():
        return False, f"2 số đầu '{prefix[:2]}' phải là số"
    
    province_code = prefix[:2]
    if province_code not in PROVINCE_MAPPING:
        return False, f"Mã tỉnh {province_code} không hợp lệ"
    
    if len(suffix_clean) < 4 or len(suffix_clean) > 5:
        return False, f"Số lượng số sau dấu - không hợp lệ ({len(suffix_clean)})"
    
    if not suffix_clean.isdigit():
        return False, f"Sau dấu - chỉ được chứa số, nhưng có '{suffix_clean}'"
    
    return True, "Hợp lệ"


def extract_province(plate_number: str) -> str:
    """Trích xuất tên tỉnh thành từ biển số"""
    if not plate_number or len(plate_number) < 2:
        return "Không xác định"
    
    province_code = plate_number[:2]
    if not province_code.isdigit():
        match = re.search(r'^([0-9]{2})', plate_number)
        if match:
            province_code = match.group(1)
        else:
            return "Không xác định"
    
    return PROVINCE_MAPPING.get(province_code, "Không xác định")


def get_plate_type(plate_number: str, is_two_lines: bool = False) -> str:
    """Xác định loại biển số"""
    if is_two_lines:
        return "motorcycle"
    
    if plate_number and '-' in plate_number:
        prefix = plate_number.split('-')[0]
        if len(prefix) == 3 and prefix[2].isdigit():
            return "motorcycle"
    
    return "car"


def validate_and_correct_plate(plate_number: str, is_two_lines: bool = False) -> str:
    """Kiểm tra và sửa biển số theo quy tắc Việt Nam"""
    if not plate_number or plate_number == "Không":
        return plate_number
    
    original = plate_number
    clean = re.sub(r'[-.]', '', plate_number)
    
    letters = [c for c in clean if c.isalpha()]
    digits = [c for c in clean if c.isdigit()]
    
    print(f"🔍 Phân tích: '{clean}'")
    print(f"   - Số chữ: {len(letters)}")
    print(f"   - Số số: {len(digits)}")
    print(f"   - Tổng: {len(clean)}")
    
    # Biển 9 số không có chữ (xe máy)
    if len(digits) == 9 and len(letters) == 0:
        print(f"⚠️ Biển 9 số không có chữ: {clean}")
        
        province = clean[:2]
        third_char = clean[2]
        rest = clean[3:]
        
        number_to_letter = {
            '0': 'D',  # D resembles 0 with a vertical line
            '1': 'L',  # L looks like 1 with a base
            '2': 'Z',  # Z is common confusion
            '3': 'E',  # E similar to 3
            '4': 'A',  # A can be confused with 4 in some fonts
            '5': 'S',  # S and 5
            '6': 'G',  # G and 6
            '7': 'T',  # T and 7
            '8': 'B',  # B and 8
            '9': 'P'   # P and 9 (or Q but Q not used)
        }
        
        letter = number_to_letter.get(third_char, 'A')
        print(f"   - Chuyển số '{third_char}' thành chữ '{letter}'")
        
        if len(rest) >= 6:
            num_type = rest[0]
            last_five = rest[1:6]
            
            if len(last_five) == 5:
                formatted = f"{province}{letter}{num_type}-{last_five[:3]}.{last_five[3:]}"
                print(f"✅ Đã sửa: '{original}' -> '{formatted}'")
                return formatted
    
    # Biển 8 số không có chữ (ô tô)
    elif len(digits) == 8 and len(letters) == 0:
        print(f"⚠️ Biển 8 số không có chữ: {clean}")
        
        province = clean[:2]
        third_char = clean[2]
        rest = clean[3:]
        
        number_to_letter = {
            '0': 'D',  # D resembles 0 with a vertical line
            '1': 'L',  # L looks like 1 with a base
            '2': 'Z',  # Z is common confusion
            '3': 'E',  # E similar to 3
            '4': 'A',  # A can be confused with 4 in some fonts
            '5': 'S',  # S and 5
            '6': 'G',  # G and 6
            '7': 'T',  # T and 7
            '8': 'B',  # B and 8
            '9': 'P'   # P and 9 (or Q but Q not used)
        }
        
        letter = number_to_letter.get(third_char, 'A')
        print(f"   - Chuyển số '{third_char}' thành chữ '{letter}'")
        
        if len(rest) >= 5:
            last_five = rest[:5]
            formatted = f"{province}{letter}-{last_five[:3]}.{last_five[3:]}"
            print(f"✅ Đã sửa: '{original}' -> '{formatted}'")
            return formatted
    
    # Đã có chữ nhưng sai vị trí
    elif len(letters) > 0:
        letter_positions = [i for i, char in enumerate(clean) if char.isalpha()]
        first_letter_pos = letter_positions[0]
        
        if first_letter_pos != 2:
            print(f"⚠️ Chữ cái ở vị trí sai: {first_letter_pos}, cần ở vị trí 2")
            
            chars = list(clean)
            first_letter = chars[first_letter_pos]
            chars.pop(first_letter_pos)
            chars.insert(2, first_letter)
            
            corrected = ''.join(chars)
            print(f"   - Sửa vị trí: '{clean}' -> '{corrected}'")
            
            if len(corrected) >= 9:
                return f"{corrected[:3]}{corrected[3]}-{corrected[4:7]}.{corrected[7:9]}"
            elif len(corrected) >= 8:
                return f"{corrected[:3]}-{corrected[3:6]}.{corrected[6:8]}"
    
    return plate_number


def postprocess_plate_result(plate_number: str, confidence: float, is_two_lines: bool = False) -> dict:
    """Hậu xử lý kết quả nhận diện biển số"""
    print(f"\n{'='*50}")
    print(f"🔧 POSTPROCESSING")
    print(f"Input plate_number: '{plate_number}'")
    print(f"is_two_lines: {is_two_lines}")
    print(f"{'='*50}")
    
    corrected = correct_ocr_errors(plate_number)
    print(f"Step 1 - Corrected: '{corrected}'")
    
    cleaned = clean_plate_numbers(corrected)
    print(f"Step 2 - Cleaned: '{cleaned}'")
    
    formatted = cleaned
    
    if '-' not in formatted:
        match = re.search(r'([A-Z0-9]+)([0-9]{4,5})', formatted)
        if match:
            prefix = match.group(1)
            numbers = match.group(2)
            if len(numbers) == 5:
                formatted = f"{prefix}-{numbers[:3]}.{numbers[3:]}"
            elif len(numbers) == 4:
                formatted = f"{prefix}-{numbers}"
    
    print(f"Step 3 - Formatted: '{formatted}'")
    
    validated = validate_and_correct_plate(formatted, is_two_lines)
    print(f"Step 4 - Validated: '{validated}'")
    
    province = extract_province(validated)
    plate_type = get_plate_type(validated, is_two_lines)
    
    print(f"📌 Kết quả cuối: {validated}")
    print(f"   - Loại: {plate_type}")
    print(f"   - Tỉnh: {province}")
    print(f"   - Độ tin cậy: {confidence:.2f}")
    
    return {
        'plate_number': validated,
        'is_valid': True,
        'validation_message': "Biển số hợp lệ",
        'plate_type': plate_type,
        'province': province,
        'final_confidence': confidence
    }