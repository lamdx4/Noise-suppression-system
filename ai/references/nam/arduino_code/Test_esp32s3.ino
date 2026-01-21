// #include <driver/i2s.h>
// #include <WiFi.h>
// #define I2S_WS 15     // LRCL
// #define I2S_SD 13     // DOUT từ mic
// #define I2S_SCK 2    // SCK

// #define SAMPLE_RATE     16000
// #define I2S_READ_LEN    512
// #define BUFFER_SIZE     I2S_READ_LEN * 4  // 32-bit = 4 byte mỗi mẫu
// const char* ssid = "Namkuner";
// const char* password = "123451211";
// const char* server_ip = "192.168.43.23";  // IP máy chạy Python server
// const uint16_t port = 3000;
// WiFiClient client;

// void setupI2S() {
//   const i2s_config_t i2s_config = {
//     .mode = i2s_mode_t(I2S_MODE_MASTER | I2S_MODE_RX),
//     .sample_rate = SAMPLE_RATE,
//     .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
//     .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
//     .communication_format = I2S_COMM_FORMAT_I2S,
//     .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
//     .dma_buf_count = 4,
//     .dma_buf_len = 256,
//     .use_apll = false,
//     .tx_desc_auto_clear = false,
//     .fixed_mclk = 0
//   };

//   const i2s_pin_config_t pin_config = {
//     .bck_io_num = I2S_SCK,
//     .ws_io_num = I2S_WS,
//     .data_out_num = I2S_PIN_NO_CHANGE,
//     .data_in_num = I2S_SD
//   };

//   i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
//   i2s_set_pin(I2S_NUM_0, &pin_config);
//   i2s_zero_dma_buffer(I2S_NUM_0);
// }

// void setup() {
//   Serial.begin(115200);
//   WiFi.begin(ssid, password);
//   Serial.print("Kết nối WiFi");
//   while (WiFi.status() != WL_CONNECTED) {
//     delay(500);
//     Serial.print(".");
//   }
//   Serial.println(" OK");

//   setupI2S();

//   Serial.println("Kết nối tới server...");
//   while (!client.connect(server_ip, port)) {
//     Serial.println("Kết nối thất bại, thử lại sau 2s...");
//     delay(2000);
//   }
//   Serial.println("Đã kết nối tới server.");
// }

// void loop() {
//   int32_t samples[I2S_READ_LEN];
//   size_t bytes_read = 0;

//   i2s_read(I2S_NUM_0, (void*)samples, sizeof(samples), &bytes_read, portMAX_DELAY);

//   // Gửi dữ liệu dạng byte
//   client.write((uint8_t*)samples, bytes_read);
//   Serial.println("Gửi xong: " + String(bytes_read) + " bytes");
// }

