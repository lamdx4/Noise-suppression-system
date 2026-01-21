#include <driver/i2s.h>
#include <WiFi.h>
#include <freertos/queue.h>

// WiFi credentials
const char* ssid = "Namkuner"; // Thay bằng SSID WiFi của bạn
const char* password = "123451211"; // Thay bằng mật khẩu WiFi của bạn

// Server details
const char* server_ip = "192.168.43.23"; // Thay bằng IP của server Python
const int server_port = 12345; // Cổng server Python

// INMP441 I2S pin configuration
#define I2S_WS 15  // Word Select (L/R)
#define I2S_SD 13  // Serial Data
#define I2S_SCK 2 // Serial Clock
#define I2S_PORT I2S_NUM_0
#define SAMPLE_RATE 8000
#define BUFFER_SIZE 500
#define QUEUE_LENGTH 16 // 8000 / 500 = 16 buffers for 1 second of audio

// FreeRTOS queue
QueueHandle_t audio_queue;
WiFiClient client;

void i2s_init() {
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT, // INMP441 mono
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 4,
    .dma_buf_len = BUFFER_SIZE,
    .use_apll = false,
  };
  
  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  
  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD
  };
  i2s_set_pin(I2S_PORT, &pin_config);
}

void task_read_i2s(void *pvParameters) {
  int16_t samples[BUFFER_SIZE];
  size_t bytes_read;

  while (1) {
    // Read audio data from I2S
    i2s_read(I2S_PORT, samples, BUFFER_SIZE * sizeof(int16_t), &bytes_read, portMAX_DELAY);
    
    if (bytes_read > 0) {
      // Send to queue
      xQueueSend(audio_queue, samples, portMAX_DELAY);
    }
    vTaskDelay(1); // Avoid watchdog timeout
  }
}

void task_send_socket(void *pvParameters) {
  int16_t samples[BUFFER_SIZE];

  while (1) {
    if (client.connected()) {
      // Receive from queue
      if (xQueueReceive(audio_queue, samples, portMAX_DELAY)) {
        // Send audio data to server
        client.write((uint8_t*)samples, BUFFER_SIZE * sizeof(int16_t));
      }
    } else {
      Serial.println("Server disconnected, attempting to reconnect...");
      client.stop();
      if (client.connect(server_ip, server_port)) {
        Serial.println("Reconnected to server");
      } else {
        Serial.println("Reconnection failed");
        vTaskDelay(5000 / portTICK_PERIOD_MS);
      }
    }
    vTaskDelay(1); // Avoid watchdog timeout
  }
}

void setup() {
  Serial.begin(115200);
  
  // Connect to WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("Connected to WiFi");

  // Connect to server
  if (client.connect(server_ip, server_port)) {
    Serial.println("Connected to server");
  } else {
    Serial.println("Connection to server failed");
  }

  // Initialize I2S
  i2s_init();

  // Create queue
  audio_queue = xQueueCreate(QUEUE_LENGTH, BUFFER_SIZE * sizeof(int16_t));

  // Create tasks
  xTaskCreate(task_read_i2s, "Read_I2S_Task", 4096, NULL, 2, NULL);
  xTaskCreate(task_send_socket, "Send_Socket_Task", 4096, NULL, 1, NULL);
}

void loop() {
  // Empty loop, tasks handle everything
  vTaskDelay(1000 / portTICK_PERIOD_MS);
}