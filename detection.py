#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File detection
"""
Module chính: Xử lý nhiều biển số, lưu ảnh các bước
ĐÃ SỬA: Thêm processed và edges vào kết quả cho từng biển số
"""

import cv2
import numpy as np
import os
import time
import re
from typing import List, Tuple, Optional, Dict, Any
from ultralytics import YOLO

from config import MODEL_PATH, YOLO_CONFIDENCE, YOLO_IOU
from utils import read_image
from preprocessing import preprocess_plate_image, crop_image_border
from ocr import recognize_plate
from postprocessing import postprocess_plate_result


class ImageProcessor:
    def __init__(self, model_path: str = MODEL_PATH, fast_mode: bool = False):
        self.model_path = model_path
        self.fast_mode = fast_mode
        self.crop_border_percent = 5
        self.yolo_model = None
        self._init_yolo()
    
    def _init_yolo(self):
        try:
            if os.path.exists(self.model_path):
                self.yolo_model = YOLO(self.model_path)
                print(f"✅ Đã tải YOLO model từ {self.model_path}")
            else:
                print(f"⚠️ Không tìm thấy model tại {self.model_path}")
        except Exception as e:
            print(f"❌ Lỗi khi tải YOLO model: {e}")
    
    def check_ready(self) -> Tuple[bool, str]:
        if self.yolo_model is None:
            return False, "YOLO model chưa được khởi tạo"
        return True, "Sẵn sàng"
    
    def detect_plate_yolo(self, image: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], bool, float, np.ndarray]:
        """Phát hiện biển số, trả về boxes, success, time, ảnh đã vẽ boxes với số thứ tự"""
        if self.yolo_model is None:
            return [], False, 0.0, image
        
        start_time = time.time()
        boxes = []
        success = False
        img_with_boxes = image.copy()
        h, w = image.shape[:2]
        
        try:
            if max(h, w) > 1280:
                scale = 1280 / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                img_resized = cv2.resize(image, (new_w, new_h))
                results = self.yolo_model(img_resized, conf=YOLO_CONFIDENCE, iou=YOLO_IOU)
                if results and len(results) > 0 and results[0].boxes is not None:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        scale_x = w / new_w
                        scale_y = h / new_h
                        x1 = int(x1 * scale_x)
                        x2 = int(x2 * scale_x)
                        y1 = int(y1 * scale_y)
                        y2 = int(y2 * scale_y)
                        x1 = max(0, x1 - 10)
                        y1 = max(0, y1 - 10)
                        x2 = min(w, x2 + 10)
                        y2 = min(h, y2 + 10)
                        boxes.append((x1, y1, x2, y2))
            else:
                results = self.yolo_model(image, conf=YOLO_CONFIDENCE, iou=YOLO_IOU)
                if results and len(results) > 0 and results[0].boxes is not None:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        x1 = max(0, x1 - 10)
                        y1 = max(0, y1 - 10)
                        x2 = min(w, x2 + 10)
                        y2 = min(h, y2 + 10)
                        boxes.append((x1, y1, x2, y2))
            
            if boxes:
                success = True
                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    text = f"Bien so {i+1}"
                    cv2.putText(img_with_boxes, text, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                print(f"🎯 Phát hiện {len(boxes)} biển số")
            else:
                print("⚠️ Không phát hiện biển số nào")
        except Exception as e:
            print(f"Lỗi detection: {e}")
        
        elapsed = time.time() - start_time
        return boxes, success, elapsed, img_with_boxes
    
    def extract_plate(self, image: np.ndarray, box: Tuple[int, int, int, int]) -> Tuple[Optional[np.ndarray], bool, str]:
        try:
            x1, y1, x2, y2 = box
            h, w = image.shape[:2]
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)
            if x1 >= x2 or y1 >= y2:
                return None, False, "Tọa độ không hợp lệ"
            plate_img = image[y1:y2, x1:x2].copy()
            if plate_img.size == 0:
                return None, False, "Ảnh cắt trống"
            return plate_img, True, "OK"
        except Exception as e:
            return None, False, f"Lỗi: {e}"
    
    def process_single_plate(self, plate_img: np.ndarray, index: int) -> Dict[str, Any]:
        """Xử lý một biển số, trả về kết quả và các ảnh trung gian"""
        result = {
            'index': index,
            'plate_number': '',
            'confidence': 0.0,
            'province_name': '',
            'success': False,
            'plate_type': '',
            'cropped_plate': None,
            'warped': None,
            'processed': None,   # Mới: ảnh sau CLAHE + sharpen (grayscale)
            'binary': None,
            'edges': None,       # Mới: ảnh cạnh Canny
            'ocr_text': ''
        }
        if plate_img is None or plate_img.size == 0:
            result['plate_number'] = "Không"
            return result
        
        try:
            result['cropped_plate'] = plate_img.copy()
            h, w = plate_img.shape[:2]
            if w < 200:
                scale = 400 / w
                new_w = int(w * scale)
                new_h = int(h * scale)
                plate_img = cv2.resize(plate_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                result['cropped_plate'] = plate_img.copy()
            
            preprocessed = preprocess_plate_image(plate_img, do_perspective=True)
            
            # Lưu tất cả các phiên bản ảnh từ preprocess
            if preprocessed.get('warped') is not None:
                result['warped'] = preprocessed['warped']
            if preprocessed.get('processed') is not None:
                # processed là ảnh grayscale, cần chuyển thành BGR để hiển thị (nếu muốn giữ màu)
                # Nhưng GUI có thể xử lý grayscale, nên để nguyên
                result['processed'] = preprocessed['processed']
            if preprocessed.get('binary') is not None:
                result['binary'] = preprocessed['binary']
            if preprocessed.get('edges') is not None:
                result['edges'] = preprocessed['edges']
            
            if preprocessed['warped'] is not None:
                h, w = preprocessed['warped'].shape[:2]
            else:
                h, w = plate_img.shape[:2]
            aspect_ratio = w / h
            is_two_lines = aspect_ratio < 1.8
            
            best_text = ""
            best_confidence = 0.0
            ocr_versions = []
            if preprocessed['warped'] is not None:
                ocr_versions.append(('warped', preprocessed['warped']))
            if preprocessed['processed'] is not None:
                # processed là grayscale, chuyển sang BGR để OCR (PaddleOCR cần 3 kênh)
                processed_bgr = cv2.cvtColor(preprocessed['processed'], cv2.COLOR_GRAY2BGR)
                ocr_versions.append(('processed', processed_bgr))
            if preprocessed['binary'] is not None:
                binary_bgr = cv2.cvtColor(preprocessed['binary'], cv2.COLOR_GRAY2BGR)
                ocr_versions.append(('binary', binary_bgr))
            
            for name, img in ocr_versions:
                if img is None: continue
                plate_text, conf = recognize_plate(None, img, None, is_two_lines)
                if plate_text and conf > best_confidence:
                    best_text = plate_text
                    best_confidence = conf
            
            if not best_text:
                plate_text, conf = recognize_plate(None, preprocessed['warped'], None, not is_two_lines)
                if plate_text:
                    best_text = plate_text
                    best_confidence = conf
            
            postprocessed = postprocess_plate_result(best_text, best_confidence, is_two_lines)
            result['plate_number'] = postprocessed['plate_number']
            result['confidence'] = postprocessed['final_confidence']
            result['province_name'] = postprocessed['province']
            result['success'] = postprocessed['is_valid']
            result['plate_type'] = postprocessed['plate_type']
            result['ocr_text'] = best_text
            
            if not result['plate_number'] or result['plate_number'] == "":
                result['plate_number'] = "Không"
                result['success'] = False
            
            if result['success']:
                print(f"   ✅ Biển {index+1}: {result['plate_number']} (conf={result['confidence']:.2f})")
            else:
                print(f"   ⚠️ Biển {index+1}: không nhận diện được")
        except Exception as e:
            result['plate_number'] = "Không"
            result['success'] = False
            print(f"   ❌ Lỗi xử lý biển {index+1}: {e}")
        return result
    
    def process_image(self, image_path: str, crop_border_percent: int = None) -> Dict[str, Any]:
        if crop_border_percent is None:
            crop_border_percent = self.crop_border_percent
        
        result = {
            'filename': os.path.basename(image_path),
            'success': False,
            'processing_time': 0.0,
            'total_plates': 0,
            'successful_plates': 0,
            'crop_border_percent': crop_border_percent,
            'original_image': None,
            'border_cropped_image': None,
            'yolo_image': None,
            'plates': []
        }
        start_time = time.time()
        
        try:
            image = read_image(image_path)
            if image is None:
                result['processing_time'] = time.time() - start_time
                return result
            result['original_image'] = image.copy()
            
            image_for_detection = image.copy()
            if crop_border_percent > 0:
                image_for_detection = crop_image_border(image_for_detection, crop_border_percent)
            result['border_cropped_image'] = image_for_detection.copy()
            
            boxes, detect_success, _, yolo_img = self.detect_plate_yolo(image_for_detection)
            result['yolo_image'] = yolo_img
            
            if not detect_success or len(boxes) == 0:
                print("⚠️ Không phát hiện biển số nào")
                result['processing_time'] = time.time() - start_time
                return result
            
            print(f"🎯 Tìm thấy {len(boxes)} biển số")
            plates_result = []
            for idx, box in enumerate(boxes):
                print(f"--- Xử lý biển số {idx+1}/{len(boxes)} ---")
                plate_img, extract_ok, msg = self.extract_plate(image_for_detection, box)
                if not extract_ok or plate_img is None:
                    plates_result.append({
                        'index': idx,
                        'plate_number': 'Không',
                        'success': False,
                        'cropped_plate': None,
                        'warped': None,
                        'processed': None,
                        'binary': None,
                        'edges': None
                    })
                    continue
                plate_result = self.process_single_plate(plate_img, idx)
                plates_result.append(plate_result)
            
            result['plates'] = plates_result
            result['total_plates'] = len(plates_result)
            result['successful_plates'] = sum(1 for p in plates_result if p.get('success', False))
            result['success'] = result['successful_plates'] > 0
            result['processing_time'] = time.time() - start_time
        except Exception as e:
            print(f"❌ Lỗi xử lý ảnh: {e}")
            result['processing_time'] = time.time() - start_time
        return result
    
    def process_batch(self, image_paths: List[str], crop_border_percent: int = 5) -> List[Dict[str, Any]]:
        results = []
        for i, path in enumerate(image_paths):
            print(f"\n[{i+1}/{len(image_paths)}] {os.path.basename(path)}")
            res = self.process_image(path, crop_border_percent)
            results.append(res)
        return results