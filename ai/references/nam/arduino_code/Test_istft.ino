// #include <arduinoFFT.h>

// #define WINDOW_SIZE 512               // Kích thước cửa sổ (phải là lũy thừa của 2)
// #define HOP_SIZE    64                // Độ chồng của các cửa sổ
// #define DATA_LENGTH 1000              // Tổng số mẫu của tín hiệu
// #define SAMPLING_FREQUENCY 1000       // Tần số lấy mẫu (Hz)
// #define FFT_BACKWARD -1
// // Số frame tính từ dữ liệu STFT
// const int numFrames = (DATA_LENGTH - WINDOW_SIZE) / HOP_SIZE + 1;

// // Mảng 2 chiều lưu STFT (phần thực và phần ảo) – giả sử đã được tính từ trước
// double fftReal[numFrames][WINDOW_SIZE];
// double fftImag[numFrames][WINDOW_SIZE];

// // Mảng lưu kết quả tái tạo tín hiệu (waveform) và tổng cửa sổ cho overlap-add
// double reconstructedSignal[DATA_LENGTH];
// double overlapSum[DATA_LENGTH];

// // Buffer dùng cho IFFT
// double ifftBufferReal[WINDOW_SIZE];
// double ifftBufferImag[WINDOW_SIZE];

// // Hàm cửa sổ Hamming (sử dụng cùng công thức với quá trình STFT)
// double hammingWindow(int n, int N) {
//   return 0.54 - 0.46 * cos((2 * PI * n) / (N - 1));
// }

// // Hàm ISTFT: tính IFFT cho mỗi frame và thực hiện overlap‑add để tái tạo tín hiệu
// void ISTFT() {
//   // Khởi tạo các mảng kết quả về 0
//   for (int i = 0; i < DATA_LENGTH; i++) {
//     reconstructedSignal[i] = 0;
//     overlapSum[i] = 0;
//   }
  
//   // Duyệt qua từng frame
//   for (int frame = 0; frame < numFrames; frame++) {
//     int startIndex = frame * HOP_SIZE;
    
//     // Copy dữ liệu STFT của frame hiện tại vào bộ đệm IFFT
//     for (int i = 0; i < WINDOW_SIZE; i++) {
//       ifftBufferReal[i] = fftReal[frame][i];
//       ifftBufferImag[i] = fftImag[frame][i];
//     }
    
//     // Tạo đối tượng ArduinoFFT và tính IFFT (chế độ FFT_BACKWARD)
//     ArduinoFFT<double> IFFT(ifftBufferReal, ifftBufferImag, WINDOW_SIZE, SAMPLING_FREQUENCY);
//     IFFT.compute(ifftBufferReal, ifftBufferImag, WINDOW_SIZE, 7, (FFTDirection)(-1));

//     // Lưu ý: Một số phiên bản của ArduinoFFT có thể tự chia kết quả cho WINDOW_SIZE.
//     // Nếu kết quả bị chia, có thể cần nhân lại với WINDOW_SIZE.
//     // Ví dụ:
//     // for (int i = 0; i < WINDOW_SIZE; i++) {
//     //   ifftBufferReal[i] *= WINDOW_SIZE;
//     // }
    
//     // Thực hiện overlap‑add: nhân cửa sổ và cộng tín hiệu vào vị trí tương ứng
//     for (int i = 0; i < WINDOW_SIZE; i++) {
//       int idx = startIndex + i;
//       if (idx < DATA_LENGTH) {
//         double windowVal = hammingWindow(i, WINDOW_SIZE);
//         reconstructedSignal[idx] += ifftBufferReal[i] * windowVal;
//         overlapSum[idx] += windowVal;
//       }
//     }
//   }
  
//   // Chuẩn hóa kết quả bằng cách chia cho tổng cửa sổ tại mỗi vị trí
//   for (int i = 0; i < DATA_LENGTH; i++) {
//     if (overlapSum[i] != 0)
//       reconstructedSignal[i] /= overlapSum[i];
//   }
// }

// void setup() {
//   Serial.begin(9600);
  
//   // ===== Giả sử dữ liệu STFT đã được tính toán trước đó =====
//   // Ở đây, chỉ khởi tạo dữ liệu mẫu cho minh họa.
//   // Ví dụ: dùng một sóng sin để tạo dữ liệu STFT giả
//   for (int frame = 0; frame < numFrames; frame++) {
//     for (int i = 0; i < WINDOW_SIZE; i++) {
//       // Tín hiệu mẫu đơn giản: sin(2π*50*t)
//       fftReal[frame][i] = sin(2 * PI * 50 * (frame * HOP_SIZE + i) / SAMPLING_FREQUENCY);
//       fftImag[frame][i] = 0;
//     }
//   }
//   // =============================================================
  
//   // Tính ISTFT để tái tạo lại tín hiệu
//   ISTFT();
  
//   // In ra một vài mẫu của tín hiệu tái tạo để kiểm tra
//   for (int i = 0; i < DATA_LENGTH; i++) {
//     Serial.println(reconstructedSignal[i]);
//   }
// }

// void loop() {
//   // Không cần xử lý thêm
// }
