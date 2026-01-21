#include <Arduino.h>
#include <WiFi.h>
#include <arduinoFFT.h>
#include <driver/i2s.h>
#include "denoi_32.h"
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// WiFi và Server
const char* ssid = "Namkuner";
const char* password = "123451211";
const char* server_ip = "192.168.43.23";
const int server_port = 3000;
unsigned long startTime,endTime,elapsedTime  ;
// Tham số STFT
#define SAMPLE_RATE 8000.0f
const int WINDOW_SIZE = 1024;
#define HOP_SIZE 256
#define BUFFER_SAMPLES 8000
#define BUFFER_COUNT 10
#define TOTAL_SAMPLES (BUFFER_SAMPLES * BUFFER_COUNT) // 80,000 mẫu = 10 giây

// I2S Pins - INMP441
#define I2S_WS 15
#define I2S_SD 13
#define I2S_SCK 2

// Khai báo cấu trúc STFTFrame
struct STFTFrame {
    float real[WINDOW_SIZE];
    float imag[WINDOW_SIZE];
};

const int NUM_FRAMES = (BUFFER_SAMPLES - WINDOW_SIZE) / HOP_SIZE + 1;

// Các biến toàn cục trong PSRAM
float (*audioBuffer)[BUFFER_SAMPLES] = nullptr;
volatile int* currentBufferCount = nullptr;
int* writeBufferIdx = nullptr;
int* writeSampleIdx = nullptr;
int* readBufferIdx = nullptr;

float* windowedReal = nullptr;
float* windowedImag = nullptr;

struct STFTFrame* stftFrames = nullptr;
struct STFTFrame* outputSTFTFrames = nullptr;

float* reconstructedSignal = nullptr;
float* windowSum = nullptr;

WiFiClient* client = nullptr;
bool* socketConnected = nullptr;

SemaphoreHandle_t* dataReadySemaphore = nullptr;
SemaphoreHandle_t* bufferMutex = nullptr;

ArduinoFFT<float>* FFT = nullptr;

tflite::MicroErrorReporter* micro_error_reporter = nullptr;
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 3 * 1024 * 1024;
uint8_t* tensor_arena = nullptr;

float hammingWindow(int n, int N) {
    return 0.54 - 0.46 * cos((2 * PI * n) / (N));
}

void initI2S() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = 64,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK,
        .ws_io_num = I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_SD
    };

    i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_NUM_0, &pin_config);
}

void initWiFi() {
    WiFi.begin(ssid, password);
    Serial.print("Đang kết nối WiFi");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nĐã kết nối WiFi");
    Serial.print("Địa chỉ IP: ");
    Serial.println(WiFi.localIP());
}

bool connectToServer() {
    Serial.printf("Kết nối đến server %s:%d...\n", server_ip, server_port);
    if (client->connect(server_ip, server_port)) {
        Serial.println("Đã kết nối đến server!");
        return true;
    } else {
        Serial.println("Kết nối thất bại!");
        return false;
    }
}

bool initTensorFlowLite() {
    model = tflite::GetModel(denoi_32);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Phiên bản schema của mô hình không phù hợp!");
        return false;
    }

    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    Serial.println("Đang cấp phát tensors...");
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        Serial.println("Cấp phát tensors thất bại!");
        return false;
    }
    Serial.println("Cấp phát tensors thành công.");

    input = interpreter->input(0);
    output = interpreter->output(0);

    Serial.print("Kích thước tensor đầu vào: ");
    for (int i = 0; i < input->dims->size; i++) {
        Serial.print(input->dims->data[i]);
        Serial.print(" ");
    }
    Serial.println();
    return true;
}

void computeSTFT(float* audioData, struct STFTFrame* frames) {
    Serial.println("Vào computeSTFT");
    for (int frame = 0; frame < NUM_FRAMES; frame++) {
        int startIndex = frame * HOP_SIZE;
        for (int i = 0; i < WINDOW_SIZE; i++) {
            windowedReal[i] = audioData[startIndex + i] * hammingWindow(i, WINDOW_SIZE);
            windowedImag[i] = 0;
        }
        FFT->compute(FFT_FORWARD);
        for (int k = 0; k < WINDOW_SIZE; k++) {
            frames[frame].real[k] = windowedReal[k];
            frames[frame].imag[k] = windowedImag[k];
        }
        if (frame % 10 == 0) taskYIELD(); // Nhường CPU sau mỗi 10 frame
    }
}

void computeISTFT(struct STFTFrame* frames, float* outputSignal) {
    memset(outputSignal, 0, BUFFER_SAMPLES * sizeof(float));
    memset(windowSum, 0, BUFFER_SAMPLES * sizeof(float));
    int fftBins = WINDOW_SIZE / 2 + 1; // 513

    float* outputData = output->data.f; // Lấy đầu ra mô hình [1, 28, 512, 2]
    int inputIdx = 0;

    for (int frame = 0; frame < NUM_FRAMES; frame++) {
        // Gán giá trị từ mô hình cho bin 0 đến 511
        for (int bin = 0; bin < fftBins - 1; bin++) { // 0 đến 511
            outputSTFTFrames[frame].real[bin] = outputData[inputIdx++]; // Real
            outputSTFTFrames[frame].imag[bin] = outputData[inputIdx++]; // Imag
        }

        // Đặt tần số Nyquist (bin 512) bằng 0
        outputSTFTFrames[frame].real[fftBins - 1] = 0.0f;
        outputSTFTFrames[frame].imag[fftBins - 1] = 0.0f;

        // Tái tạo phần đối xứng (513 đến 1023)
        for (int bin = 1; bin < fftBins - 1; bin++) {
            int mirrorBin = WINDOW_SIZE - bin;
            outputSTFTFrames[frame].real[mirrorBin] = outputSTFTFrames[frame].real[bin];
            outputSTFTFrames[frame].imag[mirrorBin] = -outputSTFTFrames[frame].imag[bin];
        }

        // Tính FFT ngược
        for (int k = 0; k < WINDOW_SIZE; k++) {
            windowedReal[k] = outputSTFTFrames[frame].real[k];
            windowedImag[k] = -outputSTFTFrames[frame].imag[k];
        }
        FFT->compute(FFT_FORWARD);
        for (int i = 0; i < WINDOW_SIZE; i++) {
            windowedReal[i] /= WINDOW_SIZE;
            windowedImag[i] = -windowedImag[i] / WINDOW_SIZE;
        }
        int startIndex = frame * HOP_SIZE;
        for (int i = 0; i < WINDOW_SIZE; i++) {
            float w = hammingWindow(i, WINDOW_SIZE);
            if (startIndex + i < BUFFER_SAMPLES) {
                outputSignal[startIndex + i] += windowedReal[i] * w;
                windowSum[startIndex + i] += w * w;
            }
        }
        if (frame % 10 == 0) taskYIELD();
    }

    for (int i = 0; i < BUFFER_SAMPLES; i++) {
        if (windowSum[i] > 1e-6) {
            outputSignal[i] /= windowSum[i];
        }
        
    }
}

void sendAudioToServer(float* audioData, int length) {
    if (!client->connected()) {
        Serial.println("Socket mất kết nối, đang thử kết nối lại...");
        *socketConnected = connectToServer();
        if (!*socketConnected) {
            Serial.println("Không thể kết nối lại, bỏ qua gửi dữ liệu.");
            return;
        }
    }

    int16_t* int16Buffer = (int16_t*)malloc(length * sizeof(int16_t));
    if (!int16Buffer) {
        Serial.println("Không thể cấp phát bộ nhớ cho buffer gửi!");
        return;
    }

    for (int i = 0; i < length; i++) {
        float sample = audioData[i];
        if (sample > 1.0f) sample = 1.0f;
        if (sample < -1.0f) sample = -1.0f;
        int16Buffer[i] = (int16_t)(sample * 32767);
    }

    size_t bytesToSend = length * sizeof(int16_t); // 16000 bytes
    const size_t CHUNK_SIZE = 1024;
    uint8_t* bufferPtr = (uint8_t*)int16Buffer;

    // Gửi header tổng
    uint32_t totalDataSize = bytesToSend;
    size_t headerSent = client->write((uint8_t*)&totalDataSize, sizeof(totalDataSize));
    if (headerSent != sizeof(totalDataSize)) {
        Serial.printf("Lỗi: Chỉ gửi được %d/%d bytes header tổng\n", headerSent, sizeof(totalDataSize));
        free(int16Buffer);
        return;
    }
    Serial.printf("Đã gửi header tổng: %d bytes\n", totalDataSize);

    // Gửi dữ liệu từng đoạn
    size_t totalBytesSent = 0;
    while (totalBytesSent < bytesToSend) {
        if (!client->connected()) {
            Serial.println("Socket mất kết nối trong khi gửi dữ liệu!");
            break;
        }

        size_t bytesRemaining = bytesToSend - totalBytesSent;
        size_t chunkSize = (bytesRemaining < CHUNK_SIZE) ? bytesRemaining : CHUNK_SIZE;

        // Gửi header đoạn
        uint32_t chunkDataSize = chunkSize;
        size_t chunkHeaderSent = client->write((uint8_t*)&chunkDataSize, sizeof(chunkDataSize));
        if (chunkHeaderSent != sizeof(chunkDataSize)) {
            Serial.printf("Lỗi: Chỉ gửi được %d/%d bytes header đoạn\n", chunkHeaderSent, sizeof(chunkDataSize));
            break;
        }
        Serial.printf("Đã gửi header đoạn: %d bytes\n", chunkDataSize);

        // Gửi dữ liệu đoạn
        size_t bytesSent = client->write(bufferPtr + totalBytesSent, chunkSize);
        if (bytesSent == 0) {
            Serial.println("Không gửi được byte nào, có thể socket đã đóng!");
            break;
        }

        totalBytesSent += bytesSent;
        Serial.printf("Gửi đoạn %d bytes, tổng cộng %d/%d bytes\n", bytesSent, totalBytesSent, bytesToSend);

        vTaskDelay(1 / portTICK_PERIOD_MS);
    }

    if (totalBytesSent == bytesToSend) {
        Serial.printf("Đã gửi thành công %d bytes dữ liệu âm thanh\n", totalBytesSent);
    } else {
        Serial.printf("Lỗi: Chỉ gửi được %d/%d bytes\n", totalBytesSent, bytesToSend);
    }

    free(int16Buffer);
}

void audioTask(void *parameter) {
    size_t bytesRead = 0;
    int samplesRead = 0;
    int32_t rawSamples[500];

    while (1) {
        xSemaphoreTake(*bufferMutex, portMAX_DELAY);
        if (*currentBufferCount == 0) {
            Serial.println("Bắt đầu thu thập 10 giây dữ liệu...");
            while (*currentBufferCount < BUFFER_COUNT) {
                i2s_read(I2S_NUM_0, rawSamples, sizeof(rawSamples), &bytesRead, portMAX_DELAY);
                samplesRead = bytesRead / sizeof(int32_t);

                if (samplesRead > 0) {
                    for (int i = 0; i < samplesRead; i++) {
                        float normalized = (float)rawSamples[i] / 2147483648.0f;
                        audioBuffer[*writeBufferIdx][*writeSampleIdx] = normalized;
                        (*writeSampleIdx)++;

                        if (*writeSampleIdx >= BUFFER_SAMPLES) {
                            *writeSampleIdx = 0;
                            (*writeBufferIdx)++;
                            (*currentBufferCount)++;
                            Serial.printf("Đã thu thập %d giây\n", *currentBufferCount);
                            if (*currentBufferCount == BUFFER_COUNT) {
                                Serial.println("Đã thu thập đủ 10 giây, báo xử lý...");
                                xSemaphoreGive(*dataReadySemaphore);
                                break;
                            }
                        }
                    }
                    vTaskDelay(1 / portTICK_PERIOD_MS); // Nhường CPU sau mỗi lần đọc I2S
                }
            }
        }
        xSemaphoreGive(*bufferMutex);
        vTaskDelay(10 / portTICK_PERIOD_MS); // Nhường CPU khi chờ audioBuffer rỗng
    }
}

void processingTask(void *parameter) {
    float* processingBuffer = (float*)ps_malloc(BUFFER_SAMPLES * sizeof(float));
    int fftBins = WINDOW_SIZE / 2 + 1; // 513

    while (1) {
        if (xSemaphoreTake(*dataReadySemaphore, portMAX_DELAY) == pdTRUE) {
            Serial.println("Đã lấy semaphore");

            while (true) {
                xSemaphoreTake(*bufferMutex, portMAX_DELAY);
                if (*currentBufferCount > 0) {
                    for (int i = 0; i < BUFFER_SAMPLES; i++) {
                        processingBuffer[i] = audioBuffer[*readBufferIdx][i];
                    }
                    for (int buf = *readBufferIdx; buf < BUFFER_COUNT - 1; buf++) {
                        memcpy(audioBuffer[buf], audioBuffer[buf + 1], BUFFER_SAMPLES * sizeof(float));
                    }
                    memset(audioBuffer[BUFFER_COUNT - 1], 0, BUFFER_SAMPLES * sizeof(float));
                    (*currentBufferCount)--;
                    Serial.printf("Đã xử lý 1 giây, còn lại %d giây\n", *currentBufferCount);

                    xSemaphoreGive(*bufferMutex);

                    Serial.println("Chuẩn bị vào computeSTFT");
                    computeSTFT(processingBuffer, stftFrames);
                    startTime = millis();
                    float* modelInput = input->data.f;

                    int inputIdx = 0;
                    for (int frame = 0; frame < NUM_FRAMES; frame++) {
                        for (int bin = 0; bin < fftBins - 1; bin++) { // 0 đến 511
                            modelInput[inputIdx++] = stftFrames[frame].real[bin];
                            modelInput[inputIdx++] = stftFrames[frame].imag[bin];
                        }
                    }

                    Serial.println("Đang chạy inference...");
                    TfLiteStatus invoke_status = interpreter->Invoke();
                    if (invoke_status != kTfLiteOk) {
                        Serial.println("Chạy inference thất bại!");
                        continue;
                    }
                    Serial.println("Inference thành công.");
                    endTime = millis();
                    elapsedTime = endTime - startTime;
                    Serial.print("Thời gian thực thi: ");
                    Serial.print(elapsedTime);
                    Serial.println(" ms");
                    // Kiểm tra kích thước đầu ra mô hình
                  Serial.print("Kích thước đầu ra của tensor (output->dims): [");
                                      for (int i = 0; i < output->dims->size; i++) {
                                          Serial.print(output->dims->data[i]);
                                          if (i < output->dims->size - 1) {
                                              Serial.print(", ");
                                          }
                                      }
                    Serial.println("]");

                    computeISTFT(outputSTFTFrames, reconstructedSignal);
                    // Serial.println("100 giá trị đầu tiên của reconstructedSignal:");
                    // for (int i = 0; i < 8000; i++) {
                    //     Serial.printf("%.4f ", reconstructedSignal[i]);
                    //     if ((i + 1) % 10 == 0) { // Xuống dòng sau mỗi 10 giá trị
                    //         Serial.println();
                    //     }
                    // }
                    if (*socketConnected) {
                        sendAudioToServer(reconstructedSignal, BUFFER_SAMPLES);
                    } else if (WiFi.status() == WL_CONNECTED) {
                        *socketConnected = connectToServer();
                    }

                    vTaskDelay(10 / portTICK_PERIOD_MS);
                } else {
                    Serial.println("audioBuffer rỗng, chờ thu thập lại...");
                    *writeBufferIdx = 0;
                    *writeSampleIdx = 0;
                    *readBufferIdx = 0;
                    xSemaphoreGive(*bufferMutex);
                    break;
                }
            }
        }
    }
}
void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("Khởi động hệ thống xử lý âm thanh ESP32...");

    // Cấp phát trong PSRAM
    audioBuffer = (float (*)[BUFFER_SAMPLES])ps_malloc(BUFFER_COUNT * BUFFER_SAMPLES * sizeof(float));
    currentBufferCount = (int*)ps_malloc(sizeof(int));
    writeBufferIdx = (int*)ps_malloc(sizeof(int));
    writeSampleIdx = (int*)ps_malloc(sizeof(int));
    readBufferIdx = (int*)ps_malloc(sizeof(int));
    windowedReal = (float*)ps_malloc(WINDOW_SIZE * sizeof(float));
    windowedImag = (float*)ps_malloc(WINDOW_SIZE * sizeof(float));
    stftFrames = (struct STFTFrame*)ps_malloc(NUM_FRAMES * sizeof(struct STFTFrame));
    outputSTFTFrames = (struct STFTFrame*)ps_malloc(NUM_FRAMES * sizeof(struct STFTFrame));
    reconstructedSignal = (float*)ps_malloc(BUFFER_SAMPLES * sizeof(float));
    windowSum = (float*)ps_malloc(BUFFER_SAMPLES * sizeof(float));
    client = (WiFiClient*)ps_malloc(sizeof(WiFiClient));
    socketConnected = (bool*)ps_malloc(sizeof(bool));
    dataReadySemaphore = (SemaphoreHandle_t*)ps_malloc(sizeof(SemaphoreHandle_t));
    bufferMutex = (SemaphoreHandle_t*)ps_malloc(sizeof(SemaphoreHandle_t));
    FFT = (ArduinoFFT<float>*)ps_malloc(sizeof(ArduinoFFT<float>));
    micro_error_reporter = (tflite::MicroErrorReporter*)ps_malloc(sizeof(tflite::MicroErrorReporter));
    tensor_arena = (uint8_t*)ps_malloc(kTensorArenaSize);

    // Khởi tạo giá trị
    *currentBufferCount = 0;
    *writeBufferIdx = 0;
    *writeSampleIdx = 0;
    *readBufferIdx = 0;
    *socketConnected = false;
    memset(audioBuffer, 0, BUFFER_COUNT * BUFFER_SAMPLES * sizeof(float));
    memset(reconstructedSignal, 0, BUFFER_SAMPLES * sizeof(float));
    memset(windowSum, 0, BUFFER_SAMPLES * sizeof(float));
    new (client) WiFiClient();
    new (FFT) ArduinoFFT<float>(windowedReal, windowedImag, WINDOW_SIZE, SAMPLE_RATE);
    new (micro_error_reporter) tflite::MicroErrorReporter();
    error_reporter = micro_error_reporter;

    if (!audioBuffer || !currentBufferCount || !writeBufferIdx || !writeSampleIdx || !readBufferIdx ||
        !windowedReal || !windowedImag || !stftFrames || !outputSTFTFrames || !reconstructedSignal ||
        !windowSum || !client || !socketConnected || !FFT || !micro_error_reporter || !tensor_arena) {
        Serial.println("Không thể cấp phát bộ nhớ trong PSRAM!");
        while (1) delay(1000);
    }

    *dataReadySemaphore = xSemaphoreCreateBinary();
    *bufferMutex = xSemaphoreCreateMutex();

    initI2S();
    initWiFi();
    *socketConnected = connectToServer();
    if (!initTensorFlowLite()) {
        Serial.println("Khởi tạo TensorFlow Lite thất bại!");
        while (1) delay(1000);
    }

    xTaskCreatePinnedToCore(audioTask, "AudioTask", 8192, NULL, 1, NULL, 0);
    xTaskCreatePinnedToCore(processingTask, "ProcessingTask", 32768, NULL, 2, NULL, 1);

    Serial.println("Hệ thống đã khởi động xong!");
}

void loop() {
    vTaskDelay(1000 / portTICK_PERIOD_MS); // Nhường CPU trong loop
}