# Nhận diện biển số xe
# Dùng phiên bản Python 3.10.11
Hệ thống nhận diện biển số xe được xây dựng bằng Python 3.10.11, đảm bảo tính ổn định và tương thích tốt với các thư viện xử lý ảnh và AI phổ biến hiện nay. 
Hệ thống sử dụng YOLO để phát hiện vị trí biển số trong ảnh, giúp xác định nhanh và chính xác vùng chứa biển số xe. 
Sau khi phát hiện, PaddleOCR được áp dụng để trích xuất ký tự từ biển số, hỗ trợ nhận diện cả biển số 1 dòng và 2 dòng. 
Cuối cùng, dữ liệu được xử lý hậu kỳ để chuẩn hóa định dạng biển số theo quy chuẩn Việt Nam, bao gồm sửa lỗi ký tự, ghép dòng và kiểm tra tính hợp lệ, giúp tăng độ chính xác của kết quả đầu ra.
