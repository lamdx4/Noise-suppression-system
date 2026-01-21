// #include <arduinoFFT.h>
// #include <math.h>  // Thêm thư viện math để sử dụng hàm sin

// #define WINDOW_SIZE 16               // Kích thước cửa sổ (phải là lũy thừa của 2)
// #define HOP_SIZE    4                // Độ chồng của các cửa sổ
// #define DATA_LENGTH 100              // Tổng số mẫu của tín hiệu
// #define SAMPLING_FREQUENCY 10.0f    // Tần số lấy mẫu (Hz)
// #define SIGNAL_FREQUENCY 2.0f         // Tần số tín hiệu sin (2Hz)

// // Các biến đã được cấp phát trong PSRAM
// double* mySignal = (double*)ps_calloc(DATA_LENGTH, sizeof(double));      // Tín hiệu đầy đủ
// double* windowedReal = (double*)ps_calloc(WINDOW_SIZE, sizeof(double));   // Dữ liệu thực của từng cửa sổ
// double* windowedImag = (double*)ps_calloc(WINDOW_SIZE, sizeof(double));  // Dữ liệu ảo của từng cửa sổ

// // Số lượng cửa sổ (frames) cho STFT
// int numFrames = (DATA_LENGTH - WINDOW_SIZE) / HOP_SIZE + 1;

// // Mảng 2 chiều lưu kết quả FFT: phần thực và phần ảo, cấp phát trong PSRAM
// double** fftReal = (double**)ps_calloc(numFrames, sizeof(double*));
// double** fftImag = (double**)ps_calloc(numFrames, sizeof(double*));

// // Khởi tạo FFT với các tham số
// ArduinoFFT<double> FFT(windowedReal, windowedImag, WINDOW_SIZE, SAMPLING_FREQUENCY);

// // Hàm cửa sổ Hamming
// double hammingWindow(int n, int N) {
//   return 0.54 - 0.46 * cos((2 * PI * n) / N);
// }

// // Hàm tính STFT
// void computeSTFT() {
//   // Cấp phát bộ nhớ cho từng hàng của mảng 2 chiều trong PSRAM
//   for (int i = 0; i < numFrames; i++) {
//     fftReal[i] = (double*)ps_calloc(WINDOW_SIZE, sizeof(double));
//     fftImag[i] = (double*)ps_calloc(WINDOW_SIZE, sizeof(double));
//   }

//   for (int frame = 0; frame < numFrames; frame++) {
//     int startIndex = frame * HOP_SIZE;
    
//     // Lấy dữ liệu của cửa sổ hiện tại và áp dụng cửa sổ Hamming
//     for (int i = 0; i < WINDOW_SIZE; i++) {
//       windowedReal[i] = mySignal[startIndex + i] * hammingWindow(i, WINDOW_SIZE);
//       windowedImag[i] = 0;
//     }
    
//     // Tính FFT cho cửa sổ này
//     FFT.compute(FFT_FORWARD);
    
//     // Sao chép kết quả vào mảng 2 chiều
//     for (int k = 0; k < WINDOW_SIZE; k++) {
//       fftReal[frame][k] = windowedReal[k];  
//       fftImag[frame][k] = windowedImag[k];  
//     }
//   }
// }

// void setup() {
//   Serial.begin(115200);

//   // Khởi tạo tín hiệu sin 2Hz
//   for (int i = 0; i < DATA_LENGTH; i++) {
//     // Tính thời gian tương ứng với mẫu thứ i
//     float t = i / SAMPLING_FREQUENCY;
//     // Tạo tín hiệu sin với tần số 2Hz
//     mySignal[i] = sin(2 * PI * SIGNAL_FREQUENCY * t);
    
//     // (Tùy chọn) Có thể thêm nhiều tần số khác nếu muốn
//     // mySignal[i] += 0.5 * sin(2 * PI * 10 * t); // Thêm thành phần 10Hz
//   }
  
//   computeSTFT();
  
//   // In kết quả
//   for (int frame = 0; frame < numFrames; frame++) {
//     Serial.print("Frame ");
//     Serial.println(frame);
//     for (int sample = 0; sample < WINDOW_SIZE; sample++) {
//       Serial.print(fftReal[frame][sample], 6); // In với 6 chữ số thập phân
//       Serial.print(" ");
//       Serial.println(fftImag[frame][sample], 6);
//     }
//     // Ngăn cách giữa các frame
//     Serial.println();
//   }

//   // Giải phóng bộ nhớ (tùy chọn)
//   /*
//   for (int i = 0; i < numFrames; i++) {
//     free(fftReal[i]);
//     free(fftImag[i]);
//   }
//   free(fftReal);
//   free(fftImag);
//   free(mySignal);
//   free(windowedReal);
//   free(windowedImag);
//   */
// }

// void loop() {
//   // Không cần xử lý thêm
// }