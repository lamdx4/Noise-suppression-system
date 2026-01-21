# Speech Enhancement
Dự án thực hiện lọc nhiễu trực tiếp qua ESP32. Có nghĩa là ESP32 tự xử lí dữ liệu và chạy model, chứ không dùng bất kì việc gọi server nào. Tuy model không có thể realtime được (có thể tối ưu thêm để realtime) nhưng đây là tiền đề để nhúng các model deeplearning vào các chíp nhúng rẻ tiền như ESP để xử lí. 
# Demo
## Âm thanh nhiễu
https://github.com/user-attachments/assets/e9d0ca52-550a-4b3e-8d2f-82887ad8a16b
## Âm thanh sau khi lọc : ở đây trong bộ dữ liệu mình chỉ train với nhiễu vỗ tay, nên những nhiễu khác không có hiệu quả.
https://github.com/user-attachments/assets/d34ff183-316d-4e37-9d69-34f83ccb781c
## Video demo gốc
[![Xem video qua youtube](https://img.youtube.com/vi/6oeu9N1jf5g/maxresdefault.jpg)](https://www.youtube.com/watch?v=6oeu9N1jf5g)
## 1. Yêu cầu phần cứng
- **ESP32**
- **Microphone I2S INM441**
- **Cáp USB để nạp code**

## 2. Yêu cầu phần mềm
- **Arduino IDE**
- **Python 3.10**
- Các thư viện cần thiết trên python:
  - `numpy`
  - `scipy`
  - `sounddevice`
  - `pydub`

## 3. Cài đặt ESP32
1. Cài đặt thư viện ESP32 trên Arduino IDE.
2. Thêm thư viện I2S và WiFi cho ESP32:
   - `ESP8266Audio`
   - `WiFi.h`
   - `Tensorflow_ESP32`
   - `arduino_FFT`
   - `TensorFlow_ESP32`
3. Flash code vào ESP32 thông qua Arduino IDE hoặc PlatformIO.

## 4. Cài đặt server Python
1. Clone repo:
   ```bash
   git clone https://github.com/namkuner/Speech-Enhancement.git
   cd Speech-Enhancement
   ```
2. Cài đặt môi trường Python:
   ```bash
   pip install -r requirements.txt
   ```
3. Chạy server:
   ```bash
   python server.py
   ```
4. Mở ArduinoIDE, gắn cáp ESP32 vào máy tính, chạy file:
   ```bash
   app.ino
   ```
## 5. Kết nối ESP32 với Server
- ESP32 sẽ gửi dữ liệu âm thanh thu được lên server thông qua WebSocket.
- Server sẽ xử lý dữ liệu và trả về kết quả đã cải thiện.

## 6. Kiểm tra hoạt động
1. Kết nối ESP32 với mạng WiFi.
2. Kiểm tra dữ liệu truyền lên server bằng cách xem log trên terminal.
3. Kiểm tra file đầu ra để xem âm thanh đã được cải thiện.

## 7. Troubleshooting
- Kiểm tra kết nối WiFi của ESP32.
- Đảm bảo server Python đang chạy.
- Kiểm tra cổng Serial Monitor trên Arduino IDE để debug lỗi ESP32.

## 8. Đóng góp
Nếu bạn có cải tiến hoặc phát hiện lỗi, vui lòng mở Pull Request hoặc Issue trên GitHub repo này.

---
Happy coding! 🚀


