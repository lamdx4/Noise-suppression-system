// #include <Arduino.h>
// #include <arduinoFFT.h>
// #include <math.h>
// #include "esp_heap_caps.h"  // Để dùng PSRAM

// // ------------------ THÔNG SỐ ------------------
// #define SAMPLE_RATE 10.0f
// #define WINDOW_SIZE 16
// #define HOP_SIZE 4
// #define BUFFER_SAMPLES 100
// const int NUM_FRAMES = ((BUFFER_SAMPLES - WINDOW_SIZE) / HOP_SIZE) + 1;

// typedef struct {
//     float real[WINDOW_SIZE];
//     float imag[WINDOW_SIZE];
// } STFTFrame;

// // ------------------ FFT ------------------
// float windowedReal[WINDOW_SIZE];
// float windowedImag[WINDOW_SIZE];
// ArduinoFFT<float> FFT = ArduinoFFT<float>(windowedReal, windowedImag, WINDOW_SIZE, SAMPLE_RATE);

// // ------------------ CỬA SỔ Hamming ------------------
// float hammingWindow(int n, int N) {
//     return 0.54f - 0.46f * cosf((2.0f * PI * n) / (N - 1));
// }

// // ------------------ STFT ------------------
// void computeSTFT(float* audioData, STFTFrame* frames) {
//     for (int frame = 0; frame < NUM_FRAMES; frame++) {
//         int startIndex = frame * HOP_SIZE;
//         for (int i = 0; i < WINDOW_SIZE; i++) {
//             windowedReal[i] = audioData[startIndex + i] * hammingWindow(i, WINDOW_SIZE);
//             windowedImag[i] = 0.0f;
//         }
//         FFT.compute(FFT_FORWARD);
//         for (int k = 0; k < WINDOW_SIZE; k++) {
//             frames[frame].real[k] = windowedReal[k];
//             frames[frame].imag[k] = windowedImag[k];
//         }
//     }
// }

// // ------------------ ISTFT ------------------
// void computeISTFT(STFTFrame* frames, float* outputSignal, float* windowSum) {
//     memset(outputSignal, 0, BUFFER_SAMPLES * sizeof(float));
//     memset(windowSum, 0, BUFFER_SAMPLES * sizeof(float));
//     for (int frame = 0; frame < NUM_FRAMES; frame++) {
//         for (int k = 0; k < WINDOW_SIZE; k++) {
//             windowedReal[k] = frames[frame].real[k];
//             windowedImag[k] = -frames[frame].imag[k];
//         }
//         FFT.compute(FFT_FORWARD);
//         for (int i = 0; i < WINDOW_SIZE; i++) {
//             windowedReal[i] /= WINDOW_SIZE;
//             windowedImag[i] = -windowedImag[i] / WINDOW_SIZE;
//         }
//         int startIndex = frame * HOP_SIZE;
//         for (int i = 0; i < WINDOW_SIZE; i++) {
//             float w = hammingWindow(i, WINDOW_SIZE);
//             if (startIndex + i < BUFFER_SAMPLES) {
//                 outputSignal[startIndex + i] += windowedReal[i] * w;
//                 windowSum[startIndex + i] += w * w;
//             }
//         }
//     }
//     for (int i = 0; i < BUFFER_SAMPLES; i++) {
//         if (windowSum[i] > 1e-6f) {
//             outputSignal[i] /= windowSum[i];
//         }
//     }
// }

// // ------------------ Tạo tín hiệu kiểm tra ------------------
// void generateTestSignal(float* signal, int length) {
//     for (int i = 0; i < length; i++) {
//         signal[i] = sinf(2.0f * PI * 1.0f * i / SAMPLE_RATE);
//     }
//     signal[50] = 2.0f;  // xung ngắn
// }

// // ------------------ In kết quả ------------------
// void printSignalSamples(float* signal, const char* name) {
//     Serial.printf("\n%s samples:\n", name);
//     Serial.println("Index\tValue");
//     for (int i = 0; i < 20; i++) {
//         Serial.printf("%d\t%.6f\n", i, signal[i]);
//     }
//     Serial.println("\nAround impulse (index 50):");
//     for (int i = 45; i < 55; i++) {
//         Serial.printf("%d\t%.6f\n", i, signal[i]);
//     }
// }

// // ------------------ SETUP ------------------
// void setup() {
//     Serial.begin(115200);
//     while (!Serial);
//     delay(1000);

//     Serial.println("\nSTFT/ISTFT Test with PSRAM Allocation");
//     Serial.printf("Sample Rate: %.1f Hz\n", SAMPLE_RATE);
//     Serial.printf("Window Size: %d\n", WINDOW_SIZE);
//     Serial.printf("Hop Size: %d\n", HOP_SIZE);
//     Serial.printf("Buffer Samples: %d\n", BUFFER_SAMPLES);
//     Serial.printf("Number of Frames: %d\n", NUM_FRAMES);

//     float* testSignal = (float*) heap_caps_malloc(BUFFER_SAMPLES * sizeof(float), MALLOC_CAP_SPIRAM);
//     float* reconstructedSignal = (float*) heap_caps_malloc(BUFFER_SAMPLES * sizeof(float), MALLOC_CAP_SPIRAM);
//     float* windowSum = (float*) heap_caps_malloc(BUFFER_SAMPLES * sizeof(float), MALLOC_CAP_SPIRAM);
//     STFTFrame* stftFrames = (STFTFrame*) heap_caps_malloc(NUM_FRAMES * sizeof(STFTFrame), MALLOC_CAP_SPIRAM);

//     if (!testSignal || !reconstructedSignal || !windowSum || !stftFrames) {
//         Serial.println("PSRAM allocation failed!");
//         while (true);
//     }

//     generateTestSignal(testSignal, BUFFER_SAMPLES);
//     printSignalSamples(testSignal, "Original");

//     Serial.println("\nComputing STFT...");
//     computeSTFT(testSignal, stftFrames);

//     Serial.println("\nComputing ISTFT...");
//     computeISTFT(stftFrames, reconstructedSignal, windowSum);
//     printSignalSamples(reconstructedSignal, "Reconstructed");

//     free(testSignal);
//     free(reconstructedSignal);
//     free(windowSum);
//     free(stftFrames);

//     Serial.println("\nTest completed!");
// }

// // ------------------ LOOP ------------------
// void loop() {
//     delay(1000);
// }
