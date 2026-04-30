#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CHƯƠNG TRÌNH CHÍNH - HỆ THỐNG NHẬN DIỆN BIỂN SỐ XE
Tích hợp YOLO + EasyOCR | Chuẩn hóa biển số Việt Nam
"""

import os
import tkinter as tk

# Import trực tiếp không cần đường dẫn
from gui import LicensePlateGUI


def main():
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║     HỆ THỐNG NHẬN DIỆN BIỂN SỐ XE                             ║
    ║     TÍCH HỢP YOLO + EASYOCR                                    ║
    ║     CHUẨN HÓA BIỂN SỐ VIỆT NAM                                 ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Hiển thị thông tin thư mục
    print(f"📁 Thư mục làm việc: {os.getcwd()}")
    print(f"📄 Các file trong thư mục:")
    for file in os.listdir():
        if file.endswith(('.py', '.pt')):
            print(f"   - {file}")
    print()
    # Kiểm tra file model
    if not os.path.exists("best.pt"):
        print("⚠️ CẢNH BÁO: Không tìm thấy file best.pt")
        print("   👉 Vui lòng tải model về và đặt cùng thư mục")
        print("   📥 Link tải: https://huggingface.co/yasirfaizahmed/license-plate-object-detection/resolve/main/best.pt\n")
    
    # Khởi tạo và chạy giao diện
    root = tk.Tk()
    
    # Center window
    root.update_idletasks()
    width = 1400
    height = 800
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    app = LicensePlateGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()