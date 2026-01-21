// #include <Arduino.h>
// #include <WiFi.h>
// #include <arduinoFFT.h>
// #include <driver/i2s.h>
// #include "denoi.h" // File chứa dữ liệu mô hình TensorFlow Lite
// #include <TensorFlowLite_ESP32.h>
// #include "tensorflow/lite/micro/all_ops_resolver.h"
// #include "tensorflow/lite/micro/micro_error_reporter.h"
// #include "tensorflow/lite/micro/micro_interpreter.h"
// #include "tensorflow/lite/schema/schema_generated.h"

// // WiFi và Server
// const char* ssid = "Namkuner";
// const char* password = "123451211";
// const char* server_ip = "192.168.43.23"; // Thay bằng IP của server Python
// const int server_port = 3000;

// // Tham số STFT
// #define SAMPLE_RATE 8000.0f          // Tần số lấy mẫu
// const int WINDOW_SIZE = 510;      // Kích thước cửa sổ NFFT
// #define HOP_SIZE 240              // Độ chồng của các cửa sổ
// #define BUFFER_SAMPLES 8000       // Số mẫu âm thanh trong 1 buffer
// #define BUFFER_COUNT 3            // Số lượng buffer
// #define TOTAL_SAMPLES (BUFFER_SAMPLES * BUFFER_COUNT)  // 9000 mẫu

// // I2S Pins - INMP441
// #define I2S_WS 15
// #define I2S_SD 13
// #define I2S_SCK 2

// // Khai báo cấu trúc STFTFrame
// struct STFTFrame {
//     float real[WINDOW_SIZE];
//     float imag[WINDOW_SIZE];
// };

// // Tính số frame STFT từ BUFFER_SAMPLES
// const int NUM_FRAMES = (BUFFER_SAMPLES - WINDOW_SIZE) / HOP_SIZE + 1;

// // Bộ đệm cho dữ liệu âm thanh
// int32_t* audioBuffer = (int32_t*)ps_calloc(TOTAL_SAMPLES, sizeof(int32_t));
// int bufferWritePos = 0;
// int bufferReadPos = 0;

// // Buffer cho biến đổi FFT
// float windowedReal[WINDOW_SIZE];
// float windowedImag[WINDOW_SIZE];

// // Mảng lưu kết quả STFT
// struct STFTFrame* stftFrames = nullptr;
// struct STFTFrame* outputSTFTFrames = nullptr;

// // Buffer cho tín hiệu đã khôi phục
// float reconstructedSignal[BUFFER_SAMPLES] = {0};
// float windowSum[BUFFER_SAMPLES] = {0};

// // Socket
// WiFiClient client;
// bool socketConnected = false;

// // Cờ báo task2 có thể xử lý dữ liệu
// SemaphoreHandle_t dataReadySemaphore;

// ArduinoFFT<float> FFT = ArduinoFFT<float>(windowedReal, windowedImag, WINDOW_SIZE, SAMPLE_RATE);

// // TensorFlow Lite
// tflite::MicroErrorReporter micro_error_reporter;
// tflite::ErrorReporter* error_reporter = &micro_error_reporter;
// static const tflite::Model* model;
// static tflite::MicroInterpreter* interpreter;
// static TfLiteTensor* input;
// static TfLiteTensor* output;

// // Tensor Arena
// constexpr int kTensorArenaSize = 1 * 1024 * 1024; // 128 KB
// static uint8_t* tensor_arena;

// // Hàm tạo cửa sổ Hamming
// float hammingWindow(int n, int N) {
//     return 0.54 - 0.46 * cos((2 * PI * n) / (N - 1));
// }

// // Khởi tạo I2S cho INMP441
// void initI2S() {
//     i2s_config_t i2s_config = {
//         .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
//         .sample_rate = SAMPLE_RATE,
//         .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
//         .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
//         .communication_format = I2S_COMM_FORMAT_STAND_I2S,
//         .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
//         .dma_buf_count = 8,
//         .dma_buf_len = 64,
//         .use_apll = false,
//         .tx_desc_auto_clear = false,
//         .fixed_mclk = 0
//     };

//     i2s_pin_config_t pin_config = {
//         .bck_io_num = I2S_SCK,
//         .ws_io_num = I2S_WS,
//         .data_out_num = I2S_PIN_NO_CHANGE,
//         .data_in_num = I2S_SD
//     };

//     i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
//     i2s_set_pin(I2S_NUM_0, &pin_config);
// }

// // Khởi tạo WiFi
// void initWiFi() {
//     WiFi.begin(ssid, password);
//     Serial.print("Đang kết nối WiFi");
//     while (WiFi.status() != WL_CONNECTED) {
//         delay(500);
//         Serial.print(".");
//     }
//     Serial.println("\nĐã kết nối WiFi");
//     Serial.print("Địa chỉ IP: ");
//     Serial.println(WiFi.localIP());
// }

// // Kết nối đến server Python
// bool connectToServer() {
//     Serial.printf("Kết nối đến server %s:%d...\n", server_ip, server_port);
    
//     if (client.connect(server_ip, server_port)) {
//         Serial.println("Đã kết nối đến server!");
//         return true;
//     } else {
//         Serial.println("Kết nối thất bại!");
//         return false;
//     }
// }

// // Khởi tạo TensorFlow Lite
// bool initTensorFlowLite() {
//     model = tflite::GetModel(denoi);
//     if (model->version() != TFLITE_SCHEMA_VERSION) {
//         Serial.println("Phiên bản schema của mô hình không phù hợp!");
//         return false;
//     }

//     static tflite::AllOpsResolver resolver;
//     static tflite::MicroInterpreter static_interpreter(
//         model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
//     interpreter = &static_interpreter;

//     Serial.println("Đang cấp phát tensors...");
//     TfLiteStatus allocate_status = interpreter->AllocateTensors();
//     if (allocate_status != kTfLiteOk) {
//         Serial.println("Cấp phát tensors thất bại!");
//         return false;
//     }
//     Serial.println("Cấp phát tensors thành công.");

//     input = interpreter->input(0);
//     output = interpreter->output(0);

//     Serial.print("Kích thước tensor đầu vào: ");
//     for (int i = 0; i < input->dims->size; i++) {
//         Serial.print(input->dims->data[i]);
//         Serial.print(" ");
//     }
//     Serial.println();

//     return true;
// }

// // Biến đổi STFT
// void computeSTFT(int32_t* audioData, struct STFTFrame* frames) {
//     Serial.println("Vào computeSTFT ");
//     for (int frame = 0; frame < NUM_FRAMES; frame++) {
//         int startIndex = frame * HOP_SIZE;
        
//         for (int i = 0; i < WINDOW_SIZE; i++) {
//             windowedReal[i] = audioData[startIndex + i] * hammingWindow(i, WINDOW_SIZE);
//             windowedImag[i] = 0;
//         }
        
//         Serial.println("Chuẩn bị vào FFT compute ");
//         FFT.compute(FFT_FORWARD);
//         printf("hoàn thành vào FFT compute");
        
//         for (int k = 0; k < WINDOW_SIZE; k++) {
//             frames[frame].real[k] = windowedReal[k];
//             frames[frame].imag[k] = windowedImag[k];
//         }
//     }
// }

// // Biến đổi ISTFT
// void computeISTFT(struct STFTFrame* frames, float* outputSignal) {
//     memset(outputSignal, 0, BUFFER_SAMPLES * sizeof(float));
//     memset(windowSum, 0, BUFFER_SAMPLES * sizeof(float));
    
//     for (int frame = 0; frame < NUM_FRAMES; frame++) {
//         int startIndex = frame * HOP_SIZE;
        
//         for (int k = 0; k < WINDOW_SIZE; k++) {
//             windowedReal[k] = frames[frame].real[k];
//             windowedImag[k] = -frames[frame].imag[k];
//         }
        
//         FFT.compute(FFT_FORWARD);
        
//         for (int i = 0; i < WINDOW_SIZE; i++) {
//             windowedReal[i] = windowedReal[i] / WINDOW_SIZE;
//             windowedImag[i] = -windowedImag[i] / WINDOW_SIZE;
//         }
        
//         for (int i = 0; i < WINDOW_SIZE; i++) {
//             float w = hammingWindow(i, WINDOW_SIZE);
//             if (startIndex + i < BUFFER_SAMPLES) {
//                 outputSignal[startIndex + i] += windowedReal[i] * w;
//                 windowSum[startIndex + i] += w * w;
//             }
//         }
//     }
    
//     for (int i = 0; i < BUFFER_SAMPLES; i++) {
//         if (windowSum[i] > 1e-6) {
//             outputSignal[i] /= windowSum[i];
//         }
//     }
// }

// // Gửi dữ liệu âm thanh qua socket
// void sendAudioToServer(float* audioData, int length) {
//     if (!client.connected()) {
//         Serial.println("Socket mất kết nối, đang thử kết nối lại...");
//         socketConnected = connectToServer();
//         if (!socketConnected) {
//             Serial.println("Kết nối lại thất bại!");
//             return;
//         }
//     }
    
//     int16_t* int16Buffer = (int16_t*)malloc(length * sizeof(int16_t));
//     if (int16Buffer == NULL) {
//         Serial.println("Không thể cấp phát bộ nhớ cho buffer gửi!");
//         return;
//     }
    
//     for (int i = 0; i < length; i++) {
//         float sample = audioData[i];
//         if (sample > 1.0f) sample = 1.0f;
//         if (sample < -1.0f) sample = -1.0f;
//         int16Buffer[i] = (int16_t)(sample * 32767);
//     }
    
//     size_t bytesToSend = length * sizeof(int16_t);
//     size_t bytesSent = client.write((uint8_t*)int16Buffer, bytesToSend);
    
//     if (bytesSent == bytesToSend) {
//         Serial.printf("Đã gửi %d bytes dữ liệu âm thanh\n", bytesSent);
//     } else {
//         Serial.printf("Lỗi: Chỉ gửi được %d/%d bytes\n", bytesSent, bytesToSend);
//     }
    
//     free(int16Buffer);
// }

// // Task 1: Thu thập dữ liệu âm thanh
// void audioTask(void *parameter) {
//     size_t bytesRead = 0;
//     int samplesRead = 0;
    
//     while (1) {
        
//         i2s_read(I2S_NUM_0, &audioBuffer[bufferWritePos], 500 * sizeof(int32_t), &bytesRead, portMAX_DELAY);
//         samplesRead = bytesRead / sizeof(int32_t);
        
//         if (samplesRead > 0) {
//             bufferWritePos += samplesRead;
            
            
//             if (bufferWritePos >= TOTAL_SAMPLES) {
//                 bufferWritePos = 0;
//             }
            
//             if ((bufferWritePos - bufferReadPos + TOTAL_SAMPLES) % TOTAL_SAMPLES >= BUFFER_SAMPLES) {
                
//                 xSemaphoreGive(dataReadySemaphore);
//             }
//         }
//         taskYIELD();
//     }
// }

// // Task 2: Xử lý dữ liệu âm thanh
// void processingTask(void *parameter) {
//     int32_t* processingBuffer = (int32_t*)ps_malloc(BUFFER_SAMPLES * sizeof(int32_t));    

//     while (1) {
//         Serial.println("Vào task 2 ");
//         if (xSemaphoreTake(dataReadySemaphore, portMAX_DELAY) == pdTRUE) {
//             Serial.println("Đã lấy semaphore");
//             for (int i = 0; i < BUFFER_SAMPLES; i++) {
//                 processingBuffer[i] = audioBuffer[(bufferReadPos + i) % TOTAL_SAMPLES];
//             }
            
//             bufferReadPos = (bufferReadPos + BUFFER_SAMPLES) % TOTAL_SAMPLES;
//             Serial.println("Chuẩn bị vào computeSTFT ");
//             computeSTFT(processingBuffer, stftFrames);
//             Serial.println("computeSTFT complete ");
//             float* modelInput = input->data.f;
//             int fftBins = WINDOW_SIZE / 2 + 1;
//             int inputIdx = 0;
            
//             for (int frame = 0; frame < NUM_FRAMES; frame++) {
//                 for (int bin = 0; bin < fftBins; bin++) {
//                     modelInput[inputIdx++] = stftFrames[frame].real[bin];
//                     modelInput[inputIdx++] = stftFrames[frame].imag[bin];
//                 }
//             }
            
//             Serial.println("Đang chạy inference...");
//             TfLiteStatus invoke_status = interpreter->Invoke();
//             if (invoke_status != kTfLiteOk) {
//                 Serial.println("Chạy inference thất bại!");
//                 continue;
//             }
//             Serial.println("Inference thành công.");
            
//             float* outputData = output->data.f;
//             inputIdx = 0;
//             for (int frame = 0; frame < NUM_FRAMES; frame++) {
//                 for (int bin = 0; bin < fftBins; bin++) {
//                     outputSTFTFrames[frame].real[bin] = outputData[inputIdx++];
//                     outputSTFTFrames[frame].imag[bin] = outputData[inputIdx++];
                    
//                     if (bin > 0 && bin < WINDOW_SIZE - bin) {
//                         outputSTFTFrames[frame].real[WINDOW_SIZE - bin] = outputSTFTFrames[frame].real[bin];
//                         outputSTFTFrames[frame].imag[WINDOW_SIZE - bin] = -outputSTFTFrames[frame].imag[bin];
//                     }
//                 }
//             }
            
//             computeISTFT(outputSTFTFrames, reconstructedSignal);
            
//             if (socketConnected) {
//                 sendAudioToServer(reconstructedSignal, BUFFER_SAMPLES);
//             } else if (WiFi.status() == WL_CONNECTED) {
//                 socketConnected = connectToServer();
//             }
//         }
//     }
// }

// void setup() {
//     Serial.begin(115200);
//     delay(1000);
//     Serial.println("Khởi động hệ thống xử lý âm thanh ESP32...");
//     Serial.printf("Heap trống trước tensor_arena: %d bytes\n", esp_get_free_heap_size());
//     tensor_arena = (uint8_t*)ps_malloc(kTensorArenaSize);
//     Serial.printf("Heap trống sau tensor_arena: %d bytes\n", esp_get_free_heap_size());
//     stftFrames = (struct STFTFrame*)ps_malloc(NUM_FRAMES * sizeof(struct STFTFrame));
//     outputSTFTFrames = (struct STFTFrame*)ps_malloc(NUM_FRAMES * sizeof(struct STFTFrame));
    
//     if (stftFrames == nullptr || outputSTFTFrames == nullptr) {
//         Serial.println("Không thể cấp phát bộ nhớ cho STFT frames!");
//         while (1) { delay(1000); }
//     }
    
//     dataReadySemaphore = xSemaphoreCreateBinary();
    
//     initI2S();
//     initWiFi();
//     socketConnected = connectToServer();
    
//     if (!initTensorFlowLite()) {
//         Serial.println("Khởi tạo TensorFlow Lite thất bại!");
//         while (1) { delay(1000); }
//     }
    
//     xTaskCreatePinnedToCore(
//         audioTask, "AudioTask", 8192, NULL, 1, NULL, 0);
//     xTaskCreatePinnedToCore(
//         processingTask, "ProcessingTask", 24576, NULL, 2, NULL, 1);
    
//     Serial.println("Hệ thống đã khởi động xong!");
// }

// void loop() {
//     // if (!socketConnected && WiFi.status() == WL_CONNECTED) {
//     //     socketConnected = connectToServer();
//     // }
//     // delay(5000);
// }

