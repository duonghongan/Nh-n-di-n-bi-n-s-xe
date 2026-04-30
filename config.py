#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#file_config
"""
Module chứa tất cả hằng số, mapping, ngưỡng và cấu hình cho hệ thống nhận diện biển số Việt Nam
"""

import re

# ==================== ĐƯỜNG DẪN ====================
MODEL_PATH = "best.pt"  # Đường dẫn model YOLO
DEFAULT_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

# ==================== NGƯỠNG XỬ LÝ ====================
# Ngưỡng YOLO
YOLO_CONFIDENCE = 0.005
YOLO_IOU = 0.15

# Ngưỡng chất lượng ảnh
MIN_PLATE_WIDTH = 20
MIN_PLATE_HEIGHT = 10
MIN_PLATE_AREA = 200
MAX_PLATE_AREA = 500000
MIN_EDGE_DENSITY = 0.05
MAX_EDGE_DENSITY = 0.8

# Ngưỡng OCR
OCR_CONFIDENCE_THRESHOLD = 0.1
MIN_CHAR_COUNT = 4
MAX_CHAR_COUNT = 12

# Ngưỡng xử lý ảnh
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150
GAUSSIAN_BLUR_KERNEL = (3, 3)
MORPH_KERNEL_SIZE = (2, 2)
SHARPEN_STRENGTH = 1.5

# ==================== BIỂN SỐ VIỆT NAM ====================
PROVINCE_MAPPING = {
    "11":"Cao Bằng","12":"Lạng Sơn","13":"Bắc Kạn","14":"Thái Nguyên","15":"Tuyên Quang",
    "16":"Hà Giang","17":"Yên Bái","18":"Lào Cai","19":"Lai Châu","20":"Điện Biên",
    "21":"Sơn La","22":"Hòa Bình","23":"Phú Thọ","24":"Vĩnh Phúc","25":"Bắc Giang",
    "26":"Bắc Ninh","27":"Quảng Ninh","28":"Hưng Yên","29":"Hà Nội","30":"Hà Nội","31":"Hà Nội",
    "32":"Hà Nội","33":"Hà Nội","34":"Hải Dương","35":"Nam Định","36":"Ninh Bình","37":"Thái Bình",
    "38":"Hà Nam","39":"Thanh Hóa","40":"Hà Nội","41":"TP Hồ Chí Minh","42":"Quảng Bình","43":"Đà Nẵng",
    "44":"Quảng Trị","45":"Thừa Thiên Huế","46":"Quảng Nam","47":"Quảng Ngãi","48":"Bình Định",
    "49":"Phú Yên","50":"TP Hồ Chí Minh","51":"TP Hồ Chí Minh","52":"TP Hồ Chí Minh","53":"TP Hồ Chí Minh",
    "54":"Khánh Hòa","55":"Ninh Thuận","56":"Bình Thuận","57":"Kon Tum","58":"Gia Lai","59":"TP Hồ Chí Minh",
    "60":"Đắk Lắk","61":"Hải Phòng","62":"Hải Phòng","63":"Hải Phòng","64":"Hải Phòng","65":"Hải Phòng",
    "66":"Đắk Nông","67":"Cần Thơ","77":"Bình Phước","78":"Tây Ninh","79":"Bình Dương","80":"Đồng Nai",
    "81":"Bà Rịa - Vũng Tàu","82":"Long An","83":"Tiền Giang","84":"Bến Tre","85":"Trà Vinh","86":"Vĩnh Long",
    "87":"Đồng Tháp","88":"An Giang","89":"Kiên Giang","90":"Hậu Giang","91":"Sóc Trăng","92":"Bạc Liêu",
    "93":"Cà Mau","94":"Cần Thơ","99":"Bắc Giang"
}

# Pattern biển số Việt Nam
VIETNAM_PLATE_PATTERN = re.compile(
    r'^([1-9][0-9])([A-Z])([0-9]?)-?([0-9]{3}\.?[0-9]{2}|[0-9]{4,5})$', re.IGNORECASE
)
CAR_PLATE_PATTERN = re.compile(
    r'^([1-9][0-9])([A-Z])-?([0-9]{3}\.?[0-9]{2}|[0-9]{4,5})$', re.IGNORECASE
)
MOTOR_PLATE_PATTERN = re.compile(
    r'^([1-9][0-9])([A-Z][0-9])-?([0-9]{3}\.?[0-9]{2}|[0-9]{4,5})$', re.IGNORECASE
)

VALID_CHARS = set('0123456789ABCDEFGHJKLMNPQRSTUVWXYZ')
OCR_CORRECTIONS = {'8':'B','0':'O','1':'I','5':'S','2':'Z'}

# ==================== KÍCH THƯỚC BIỂN SỐ ====================
# 1 dòng
WARPED_PLATE_WIDTH = 240
WARPED_PLATE_HEIGHT = 80

# 2 dòng
WARPED_PLATE_WIDTH_2_LINE = 240
WARPED_PLATE_HEIGHT_2_LINE = 140

# ==================== ĐỊNH DẠNG XUẤT FILE ====================
EXCEL_COLUMNS = ['STT', 'Tên file ảnh', 'Biển số xe', 'Loại xe', 'Tỉnh thành', 'Trạng thái', 'Độ tin cậy', 'Thời gian xử lý (s)']
CSV_ENCODING = 'utf-8-sig'

# ==================== CẤU HÌNH OCR ====================
# Chỉ sử dụng PaddleOCR
OCR_ENGINE = 'paddleocr'  # Chỉ dùng PaddleOCR

# Cấu hình PaddleOCR
PADDLEOCR_LANGUAGES = 'en'  # 'en' cho tiếng Anh, 'vietnam' cho tiếng Việt (nếu có)
PADDLEOCR_USE_GPU = False   # Đặt True nếu có GPU

# Cấu hình cũ (giữ lại để tương thích nhưng không dùng)
EASYOCR_LANGUAGES = ['en']
EASYOCR_GPU = False
EASYOCR_MODEL_STORAGE_DIRECTORY = './easyocr_model'

# ==================== CẤU HÌNH HIỂN THỊ GUI ====================
DISPLAY_SIZE = (500, 350)
SMALL_DISPLAY_SIZE = (300, 200)