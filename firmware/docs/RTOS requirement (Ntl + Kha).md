- Yêu cầu: Dùng platformIO extension VSCode để code ko dùng Aduino, code trong repo: [lamdx4/Noise-suppression-system](https://github.com/lamdx4/Noise-suppression-system). Code được đặt trong folder firmware  
- Task AI tạm thời chưa có nhưng vẫn phải triển khai và phải giả lập độ trễ 10-100ms và do chưa có AI nên đầu vào thu âm gì cứ phát đầu ra loa như thế luôn nhưng vẫn phải quá task AI (Giả lập)

### **1\. Bảng Cấu Hình Task (Task Configuration Table)**

| Tên Task | Core Pinning | Độ ưu tiên (Priority) | Stack Size (Min) | Tần suất / Ràng buộc (Timing) | Chức năng cụ thể & Lưu ý |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **task\_i2s\_read** (Thu Âm) | **Core 0** (Protocol/IO) | **Cao (20)** *(Cao hơn Process)* | 4 KB | **Liên tục** Chờ DMA ngắt | \- Đọc DMA I2S (Block chờ). \- Convert int16 sang float. \- **QUAN TRỌNG:** Ghi vào *Ring Buffer* ngay lập tức. Không được làm gì nặng ở đây. |
| **task\_rnnoise** (Xử lý AI) | **Core 1** (App CPU) | **Trung (10)** *(Thấp hơn I/O)* | **10 \- 12 KB** *(Rất lớn)* | **\< 10ms / Frame** (Deadline cứng) | \- Lấy data từ *Ring Buffer Input*. \- Chạy rnnoise\_process\_frame. \- Ghi kết quả vào *Ring Buffer Output*. \- **Lưu ý:** Task này sẽ ngốn 100% Core 1 khi chạy. |
| **task\_i2s\_write** (Phát Loa) | **Core 0** (Protocol/IO) | **Cao (20)** *(Ngang task Thu)* | 4 KB | **Liên tục** Đẩy ra DAC | \- Lấy data từ *Ring Buffer Output*. \- Convert float về int16. \- Ghi ra I2S/DAC hoặc gửi WiFi (UDP). |

### 

### **2\. Các Component Giao Tiếp (Inter-Process Communication)**

Thay vì dùng Queue thường (nặng nề), chúng ta dùng **Ring Buffer** (hoặc Stream Buffer) để đạt hiệu suất "Zero-copy" hoặc tốc độ cao.

| Component | Loại (Type) | Kích thước (Size) | Vai trò Thực tế |
| :---- | :---- | :---- | :---- |
| **rb\_in** | **Ring Buffer** (xRingbufferCreate) | **2048 \- 4096 bytes** *(\~4-8 frames)* | **Jitter Buffer Đầu Vào:** \- Chứa dữ liệu thô từ Mic. \- Nếu Task AI bị chậm 1 nhịp (do vướng ngắt), buffer này sẽ phình to ra để chứa hộ, không làm mất mẫu. |
| **rb\_out** | **Ring Buffer** (xRingbufferCreate) | **2048 \- 4096 bytes** *(\~4-8 frames)* | **Jitter Buffer Đầu Ra:** \- Chứa dữ liệu sạch. \- Task Phát sẽ luôn có sẵn hàng để đẩy ra loa, tránh việc loa bị "khực" (underrun) khi Task AI chưa tính xong. |
| **mutex\_model** | **Semaphore** | 1 | **Bảo vệ Model (Optional):** \- Nếu bạn có chức năng chỉnh thông số Gain/Threshold từ App điện thoại trong lúc đang chạy, cần Mutex này để tránh crash khi 2 luồng cùng truy cập struct DenoiseState. |

### 

### **3\. Flow Dữ Liệu Thực Tế (Data Flow)**

Một gói tin âm thanh (Frame 480 mẫu) sẽ đi qua "dây chuyền sản xuất" như sau:

1. **Thu (Capture):**  
   * Phần cứng I2S tự động đổ đầy DMA Buffer.  
   * task\_i2s\_read tỉnh dậy, copy dữ liệu từ DMA \-\> rb\_in.  
   * *Thời gian thực hiện:* Cực nhanh (\~0.1ms).  
2. **Đệm (Buffering):**  
   * Dữ liệu nằm chờ trong rb\_in. Nếu CPU đang rảnh, nó được lấy ngay. Nếu CPU bận, nó nằm đó chờ (an toàn trong khoảng 4-8 frame \~ 40-80ms).  
3. **Xử lý (Processing \- Core 1):**  
   * task\_rnnoise dùng hàm xRingbufferReceive(..., pdMS\_TO\_TICKS(20)) để chờ.  
   * Nhận được 1 Frame \-\> Gọi rnnoise\_process\_frame.  
   * Tính toán mất khoảng **8ms \- 9ms** (trên ESP32 240MHz).  
   * Hiện tại chưa có nên giả lập độ trễ là 10ms \-\> 100ms  
   * Kết quả đẩy xuống rb\_out.  
4. **Phát (Playback \- Core 0):**  
   * task\_i2s\_write liên tục moi dữ liệu từ rb\_out.  
   * Nếu rb\_out trống (AI tính chưa xong) \-\> Ghi Silence (Im lặng) hoặc lặp lại mẫu cuối (để giấu lỗi).  
   * Đẩy ra I2S DAC.

### **4\. Yêu Cầu Hệ Thống Thực Tế (System Requirements)**

Để hệ thống này chạy "mượt như Sunsilk", cần đảm bảo các yêu cầu phần cứng sau:

* **CPU Clock:** Bắt buộc **240 MHz**. (Mặc định 160 MHz sẽ không kịp deadline 10ms).  
* **RAM:** Cần dư khoảng **50 KB Heap** để khởi tạo RNNoise (DenoiseState \+ các buffer tạm của nó) và Stack cho Task.  
  * *Mẹo:* Nếu thiếu RAM, hãy bật PSRAM (nếu board có hỗ trợ) và cấp phát các buffer to vào đó.  
* **Flash:** Code RNNoise khá nặng (do chứa bảng rnn\_model), cần khoảng **1MB \- 2MB** Flash trống cho App partition.  
* **Latency (Độ trễ toàn mạch):**  
  * Tổng trễ \= (Size rb\_in) \+ (Thời gian tính toán) \+ (Size rb\_out).  
  * Với cấu hình trên, độ trễ âm thanh từ lúc nói vào mic đến lúc nghe thấy ở loa sẽ khoảng **40ms \- 60ms**. Đây là mức chấp nhận được cho giao tiếp thời gian thực (như bộ đàm).