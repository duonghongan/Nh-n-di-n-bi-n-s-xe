#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file gui - Xuất Excel 2 cột (Filename, Biensoxe)
"""
Giao diện đồ họa - 9 tab, chỉ hiển thị biển số hợp lệ
Xuất file Excel: 2 cột, biển sạch, nhiều biển cách nhau ||
"""

import os
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import threading
import pandas as pd
from datetime import datetime
import cv2
import numpy as np
from PIL import Image, ImageTk
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from detection import ImageProcessor
from postprocessing import validate_plate_format


class LicensePlateGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("HỆ THỐNG NHẬN DIỆN BIỂN SỐ XE - Xuất Excel 2 cột")
        self.root.geometry("1600x900")
        self.root.configure(bg='#f0f0f0')
        
        self.processor = None
        self.fast_mode = tk.BooleanVar(value=False)
        self.crop_border_enabled = tk.BooleanVar(value=True)
        self.crop_border_percent = tk.IntVar(value=5)
        self.init_processor()
        
        self.image_files = []
        self.results = []
        self.current_result = None
        self.current_plate_idx = 0
        self.stop_flag = False
        self.executor = None
        
        self.setup_ui()
        self.update_status()
    
    def init_processor(self):
        try:
            self.processor = ImageProcessor("best.pt", fast_mode=self.fast_mode.get())
            ready, msg = self.processor.check_ready()
            if not ready:
                messagebox.showwarning("Cảnh báo", f"Processor chưa sẵn sàng: {msg}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể khởi tạo: {e}")
    
    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        tk.Label(header_frame, text="🔍 NHẬN DIỆN BIỂN SỐ XE - Xuất Excel 2 cột",
                 font=('Arial', 16, 'bold'), bg='#2c3e50', fg='white').pack(expand=True)
        
        main_container = tk.Frame(self.root, bg='#f0f0f0')
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Cột trái
        left_frame = tk.Frame(main_container, bg='white', relief='solid', borderwidth=1, width=350)
        left_frame.pack(side='left', fill='both', expand=False, padx=5)
        left_frame.pack_propagate(False)
        
        tk.Label(left_frame, text="📁 NHẬP ẢNH", font=('Arial', 12, 'bold'), bg='white').pack(pady=10)
        btn_frame = tk.Frame(left_frame, bg='white')
        btn_frame.pack(pady=5)
        self.btn_select = tk.Button(btn_frame, text="📂 Chọn ảnh", command=self.select_images,
                                    bg='#3498db', fg='white', font=('Arial',10,'bold'), width=12)
        self.btn_select.pack(side='left', padx=5)
        self.btn_clear = tk.Button(btn_frame, text="🗑️ Xóa all", command=self.clear_images,
                                   bg='#e74c3c', fg='white', font=('Arial',10,'bold'), width=12)
        self.btn_clear.pack(side='left', padx=5)
        
        tk.Checkbutton(left_frame, text="⚡ Chế độ xử lý nhanh", variable=self.fast_mode,
                       command=self.toggle_fast_mode, bg='white').pack(pady=5)
        
        crop_frame = tk.Frame(left_frame, bg='white')
        crop_frame.pack(pady=5)
        self.crop_check = tk.Checkbutton(crop_frame, text="✂️ Cắt viền TRƯỚC detection",
                                         variable=self.crop_border_enabled, bg='white',
                                         command=self.toggle_crop_border)
        self.crop_check.pack(side='left')
        self.crop_spin = tk.Spinbox(crop_frame, from_=1, to=15, textvariable=self.crop_border_percent,
                                    width=5, command=self.update_crop_percent)
        self.crop_spin.pack(side='left', padx=5)
        tk.Label(crop_frame, text="%", bg='white').pack(side='left')
        
        tk.Label(left_frame, text="Danh sách ảnh đã chọn:", bg='white', anchor='w').pack(fill='x', padx=10, pady=(10,5))
        list_frame = tk.Frame(left_frame, bg='white')
        list_frame.pack(fill='both', expand=True, padx=10, pady=5)
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side='right', fill='y')
        self.listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, font=('Arial',10))
        self.listbox.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.listbox.yview)
        self.stats_label = tk.Label(left_frame, text="📊 Số lượng: 0 ảnh", font=('Arial',10,'bold'), bg='white', fg='#27ae60')
        self.stats_label.pack(pady=10)
        
        # Cột giữa: 9 tab
        middle_frame = tk.Frame(main_container, bg='white', relief='solid', borderwidth=1, width=700)
        middle_frame.pack(side='left', fill='both', expand=True, padx=5)
        middle_frame.pack_propagate(False)
        
        select_frame = tk.Frame(middle_frame, bg='white')
        select_frame.pack(fill='x', padx=5, pady=5)
        tk.Label(select_frame, text="Chọn biển số:", bg='white').pack(side='left')
        self.plate_combo = ttk.Combobox(select_frame, state='readonly', width=20)
        self.plate_combo.pack(side='left', padx=5)
        self.plate_combo.bind('<<ComboboxSelected>>', self.on_plate_select)
        
        self.notebook = ttk.Notebook(middle_frame)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.tabs = {}
        tab_config = [
            ("📷 1. Ảnh gốc", "original"),
            ("✂️ 2. Sau cắt viền", "border_cropped"),
            ("🎯 3. YOLO detection", "yolo"),
            ("✂️ 4. Biển số đã cắt", "cropped"),
            ("📐 5. Ảnh cạnh", "edges"),
            ("🔄 6. Đã chỉnh góc", "warped"),
            ("✨ 7. Ảnh tăng cường", "processed"),
            ("⚫ 8. Ảnh nhị phân", "binary"),
            ("🔤 9. Kết quả OCR", "ocr")
        ]
        for text, name in tab_config:
            frame = tk.Frame(self.notebook, bg='white')
            self.notebook.add(frame, text=text)
            self.tabs[name] = frame
            canvas = tk.Canvas(frame, bg='#ecf0f1', relief='sunken')
            canvas.pack(fill='both', expand=True, padx=5, pady=5)
            label = tk.Label(frame, text="", bg='white')
            label.pack()
            setattr(self, f'canvas_{name}', canvas)
            setattr(self, f'label_{name}', label)
        
        ocr_frame = self.tabs["ocr"]
        for widget in ocr_frame.winfo_children():
            widget.destroy()
        self.ocr_text = ScrolledText(ocr_frame, font=('Consolas', 11), bg='#f8f9fa')
        self.ocr_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Cột phải
        right_frame = tk.Frame(main_container, bg='white', relief='solid', borderwidth=1, width=400)
        right_frame.pack(side='left', fill='both', expand=False, padx=5)
        right_frame.pack_propagate(False)
        
        tk.Label(right_frame, text="⚙️ XỬ LÝ", font=('Arial',12,'bold'), bg='white').pack(pady=10)
        process_frame = tk.Frame(right_frame, bg='white')
        process_frame.pack(pady=10)
        self.btn_process = tk.Button(process_frame, text="▶️ BẮT ĐẦU XỬ LÝ", command=self.process_images,
                                     bg='#27ae60', fg='white', font=('Arial',11,'bold'), width=18, height=2)
        self.btn_process.pack(side='left', padx=5)
        self.btn_stop = tk.Button(process_frame, text="⏹️ DỪNG", command=self.stop_processing,
                                  bg='#e74c3c', fg='white', state='disabled', width=10, height=2)
        self.btn_stop.pack(side='left', padx=5)
        
        self.progress = ttk.Progressbar(right_frame, length=300, mode='determinate')
        self.progress.pack(pady=5)
        
        tk.Label(right_frame, text="Kết quả xử lý:", font=('Arial',10,'bold'), bg='white').pack(pady=(15,5))
        self.summary_text = ScrolledText(right_frame, height=15, width=40, font=('Consolas',9))
        self.summary_text.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.btn_export = tk.Button(right_frame, text="📥 XUẤT EXCEL", command=self.export_to_excel,
                                    bg='#f39c12', fg='white', font=('Arial',10,'bold'), width=15)
        self.btn_export.pack(pady=10)
        
        self.listbox.bind('<<ListboxSelect>>', self.on_image_select)
    
    def toggle_fast_mode(self):
        if self.processor:
            self.processor.fast_mode = self.fast_mode.get()
    
    def toggle_crop_border(self):
        self.crop_spin.config(state='normal' if self.crop_border_enabled.get() else 'disabled')
    
    def update_crop_percent(self):
        pass
    
    def select_images(self):
        files = filedialog.askopenfilenames(title="Chọn ảnh", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        added = 0
        for f in files:
            if f not in self.image_files:
                self.image_files.append(f)
                self.listbox.insert(tk.END, os.path.basename(f))
                added += 1
        self.update_status()
        if added:
            messagebox.showinfo("Thông báo", f"Đã thêm {added} ảnh")
    
    def clear_images(self):
        if messagebox.askyesno("Xác nhận", "Xóa tất cả ảnh?"):
            self.image_files = []
            self.results = []
            self.listbox.delete(0, tk.END)
            self.clear_display()
            self.update_status()
            self.summary_text.delete(1.0, tk.END)
    
    def clear_display(self):
        for name in ['original', 'border_cropped', 'yolo', 'cropped', 'edges', 'warped', 'processed', 'binary']:
            canvas = getattr(self, f'canvas_{name}', None)
            if canvas:
                canvas.delete("all")
            label = getattr(self, f'label_{name}', None)
            if label:
                label.config(text="")
        if hasattr(self, 'ocr_text'):
            self.ocr_text.delete(1.0, tk.END)
        self.plate_combo.set('')
        self.plate_combo['values'] = []
    
    def update_status(self):
        self.stats_label.config(text=f"📊 Số lượng: {len(self.image_files)} ảnh")
    
    def on_image_select(self, event):
        selection = self.listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        if idx < len(self.image_files):
            self.current_image_path = self.image_files[idx]
            self.current_result = None
            for res in self.results:
                if res.get('filename') == os.path.basename(self.current_image_path):
                    self.current_result = res
                    break
            self.current_plate_idx = 0
            self.update_display()
    
    def on_plate_select(self, event):
        if self.plate_combo.current() >= 0:
            self.current_plate_idx = self.plate_combo.current()
            self.update_display()
    
    def get_successful_plates(self, result):
        plates = result.get('plates', [])
        successful = []
        for p in plates:
            if p.get('success', False):
                plate_num = p.get('plate_number', '')
                if plate_num and plate_num != "Không":
                    is_valid, _ = validate_plate_format(plate_num)
                    if is_valid:
                        successful.append(p)
        return successful
    
    def update_display(self):
        self.clear_display()
        if not self.current_result:
            return
        plates = self.get_successful_plates(self.current_result)
        if plates:
            plate_names = [f"Biển {i+1}: {p['plate_number']}" for i, p in enumerate(plates)]
            self.plate_combo['values'] = plate_names
            if self.current_plate_idx < len(plate_names):
                self.plate_combo.set(plate_names[self.current_plate_idx])
            else:
                self.current_plate_idx = 0
                if plate_names:
                    self.plate_combo.set(plate_names[0])
        else:
            self.plate_combo['values'] = []
            self.plate_combo.set('')
        
        self.display_image_on_canvas(self.current_result.get('original_image'), 'original', "Ảnh gốc")
        self.display_image_on_canvas(self.current_result.get('border_cropped_image'), 'border_cropped', "Sau cắt viền")
        self.display_image_on_canvas(self.current_result.get('yolo_image'), 'yolo', "YOLO detection")
        
        if plates and self.current_plate_idx < len(plates):
            plate = plates[self.current_plate_idx]
            self.display_image_on_canvas(plate.get('cropped_plate'), 'cropped', f"Biển số {self.current_plate_idx+1} đã cắt")
            self.display_image_on_canvas(plate.get('edges'), 'edges', "Ảnh cạnh")
            self.display_image_on_canvas(plate.get('warped'), 'warped', "Đã chỉnh góc")
            self.display_image_on_canvas(plate.get('processed'), 'processed', "Ảnh tăng cường")
            self.display_image_on_canvas(plate.get('binary'), 'binary', "Ảnh nhị phân")
            self.ocr_text.delete(1.0, tk.END)
            self.ocr_text.insert(tk.END, f"🔢 Biển số: {plate['plate_number']}\n")
            self.ocr_text.insert(tk.END, f"📊 Độ tin cậy: {plate['confidence']:.2f}%\n")
            self.ocr_text.insert(tk.END, f"🗺️ Tỉnh/TP: {plate['province_name']}\n")
        else:
            for name in ['cropped', 'edges', 'warped', 'processed', 'binary']:
                self.display_image_on_canvas(None, name, "Không có biển số")
            self.ocr_text.insert(tk.END, "Không có biển số nào được nhận diện")
    
    def display_image_on_canvas(self, image, name, text):
        canvas = getattr(self, f'canvas_{name}', None)
        label = getattr(self, f'label_{name}', None)
        if canvas is None or label is None:
            return
        canvas.delete("all")
        if image is None:
            label.config(text=f"{text}: Không có ảnh")
            return
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif len(image.shape) == 2:
                rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                rgb = image
            pil_img = Image.fromarray(rgb)
        else:
            pil_img = image
        display_size = (500, 350)
        pil_img.thumbnail(display_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(pil_img)
        canvas.config(width=display_size[0], height=display_size[1])
        canvas.create_image(display_size[0]//2, display_size[1]//2, image=photo, anchor='center')
        canvas.image = photo
        label.config(text=f"{text} ({pil_img.width}x{pil_img.height})")
    
    def process_images(self):
        if not self.image_files:
            messagebox.showwarning("Cảnh báo", "Chưa chọn ảnh!")
            return
        self.btn_process.config(state='disabled')
        self.btn_stop.config(state='normal')
        self.btn_export.config(state='disabled')
        self.summary_text.delete(1.0, tk.END)
        self.progress['maximum'] = len(self.image_files)
        self.progress['value'] = 0
        self.stop_flag = False
        self.start_time = time.time()
        thread = threading.Thread(target=self.process_thread)
        thread.daemon = True
        thread.start()
    
    def stop_processing(self):
        if messagebox.askyesno("Xác nhận", "Dừng xử lý?"):
            self.stop_flag = True
            if self.executor:
                self.executor.shutdown(wait=False)
            self.btn_stop.config(state='disabled')
    
    def process_thread(self):
        total = len(self.image_files)
        crop_percent = self.crop_border_percent.get() if self.crop_border_enabled.get() else 0
        self.results = [None] * total
        with ThreadPoolExecutor(max_workers=4) as executor:
            self.executor = executor
            future_to_idx = {}
            for i, path in enumerate(self.image_files):
                if self.stop_flag:
                    break
                future = executor.submit(self.processor.process_image, path, crop_percent)
                future_to_idx[future] = i
            for future in as_completed(future_to_idx):
                if self.stop_flag:
                    break
                idx = future_to_idx[future]
                try:
                    res = future.result()
                    self.results[idx] = res
                except Exception as e:
                    self.results[idx] = {'filename': os.path.basename(self.image_files[idx]), 'success': False, 'plates': []}
                self.root.after(0, self.update_progress, idx+1)
                self.root.after(0, self.display_summary)
        if not self.stop_flag:
            self.end_time = time.time()
            self.root.after(0, self.finish_processing)
        else:
            self.root.after(0, self.enable_buttons)
    
    def update_progress(self, val):
        self.progress['value'] = val
    
    def display_summary(self):
        self.summary_text.delete(1.0, tk.END)
        total_success_plates = 0
        total_images_with_plates = 0
        for res in self.results:
            if res is None:
                continue
            filename = res.get('filename', 'unknown')
            successful_plates = self.get_successful_plates(res)
            num_success = len(successful_plates)
            if num_success > 0:
                total_success_plates += num_success
                total_images_with_plates += 1
                self.summary_text.insert(tk.END, f"✅ {filename}: {num_success} biển\n")
                for i, plate in enumerate(successful_plates):
                    plate_num = plate.get('plate_number', '')
                    province = plate.get('province_name', '')
                    self.summary_text.insert(tk.END, f"   Biển số {i+1}: {plate_num} - {province}\n")
                self.summary_text.insert(tk.END, "\n")
            else:
                self.summary_text.insert(tk.END, f"❌ {filename}: Không có biển số hợp lệ\n\n")
        self.summary_text.insert(tk.END, f"📊 Tổng kết: {total_success_plates} biển hợp lệ trên {total_images_with_plates} ảnh\n")
    
    def finish_processing(self):
        self.enable_buttons()
        total_time = time.time() - self.start_time
        self.summary_text.insert(tk.END, f"⏱️ Tổng thời gian: {total_time:.2f}s")
        self.on_image_select(None)
    
    def enable_buttons(self):
        self.btn_process.config(state='normal')
        self.btn_stop.config(state='disabled')
        self.btn_export.config(state='normal')
    
    def export_to_excel(self):
        """Xuất Excel 2 cột: Filename, Biensoxe (biển sạch, nhiều biển cách ||)"""
        if not self.results:
            messagebox.showwarning("Cảnh báo", "Chưa có kết quả!")
            return
        filepath = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            initialfile=f"ket_qua_bien_so_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )
        if not filepath:
            return

        data_rows = []
        for res in self.results:
            if res is None:
                continue
            filename = res.get('filename', '')
            successful_plates = self.get_successful_plates(res)
            clean_plates = []
            for plate in successful_plates:
                plate_num = plate.get('plate_number', '')
                # Chỉ giữ chữ cái và số
                clean = re.sub(r'[^A-Za-z0-9]', '', plate_num)
                if clean:
                    clean_plates.append(clean)
            biensoxe = '||'.join(clean_plates)
            data_rows.append([filename, biensoxe])

        df = pd.DataFrame(data_rows, columns=['Filename', 'Biensoxe'])
        df.to_excel(filepath, index=False)
        messagebox.showinfo("Thành công", f"✅ Đã xuất {len(data_rows)} ảnh ra file Excel:\n{filepath}")
    
    def on_closing(self):
        if self.executor:
            self.executor.shutdown(wait=False)
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = LicensePlateGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()