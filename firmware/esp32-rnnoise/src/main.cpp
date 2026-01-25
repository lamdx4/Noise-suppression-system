#include "driver/i2s_std.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
#include "lwip/err.h"
#include "lwip/sockets.h"
#include "lwip/sys.h"
#include "nvs_flash.h"
#include <string.h>

#include "constants/i2s_config_t.h"

// =============================================================
// CONFIGURATION
// =============================================================
#define WIFI_SSID "J192"
#define WIFI_PASS "hoangchimbe"
#define PC_IP_ADDR "192.168.43.223" // Hãy đổi thành IP máy tính hiện tại
#define PC_PORT 12345

static const char *TAG = "AUDIO";
static EventGroupHandle_t s_wifi_event_group;
#define WIFI_CONNECTED_BIT BIT0

static void event_handler(void *arg, esp_event_base_t event_base,
                          int32_t event_id, void *event_data) {
  if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
    esp_wifi_connect();
  } else if (event_base == WIFI_EVENT &&
             event_id == WIFI_EVENT_STA_DISCONNECTED) {
    esp_wifi_connect();
    ESP_LOGI(TAG, "Retrying to connect to the AP");
  } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
    ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
    ESP_LOGI(TAG, "Got IP:" IPSTR, IP2STR(&event->ip_info.ip));
    xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
  }
}

void wifi_init_sta(void) {
  s_wifi_event_group = xEventGroupCreate();
  ESP_ERROR_CHECK(esp_netif_init());
  ESP_ERROR_CHECK(esp_event_loop_create_default());
  esp_netif_create_default_wifi_sta();

  wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
  ESP_ERROR_CHECK(esp_wifi_init(&cfg));

  esp_event_handler_instance_t instance_any_id;
  esp_event_handler_instance_t instance_got_ip;
  ESP_ERROR_CHECK(esp_event_handler_instance_register(
      WIFI_EVENT, ESP_EVENT_ANY_ID, &event_handler, NULL, &instance_any_id));
  ESP_ERROR_CHECK(esp_event_handler_instance_register(
      IP_EVENT, IP_EVENT_STA_GOT_IP, &event_handler, NULL, &instance_got_ip));

  wifi_config_t wifi_config = {
      .sta = {
          .ssid = WIFI_SSID,
          .password = WIFI_PASS,
          .threshold = {.authmode = WIFI_AUTH_WPA2_PSK},
      },
  };
  ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
  ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
  ESP_ERROR_CHECK(esp_wifi_start());
  
  // FIX 1: Tắt tiết kiệm pin để Wifi gửi nhanh hơn
  ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));

  // FIX 2: Giảm công suất phát Wifi để giảm nhiễu rè vào Mic
  esp_wifi_set_max_tx_power(52); // Mức trung bình (khoảng 13dBm)

  ESP_LOGI(TAG, "Wi-Fi initialization finished. Waiting for connection...");
  xEventGroupWaitBits(s_wifi_event_group, WIFI_CONNECTED_BIT, pdFALSE, pdFALSE,
                      portMAX_DELAY);
  
  // In ra IP để kiểm tra
  esp_netif_ip_info_t ip_info;
  esp_netif_get_ip_info(esp_netif_get_handle_from_ifkey("WIFI_STA_DEF"), &ip_info);
  ESP_LOGI(TAG, "Connected! ESP32 IP: " IPSTR, IP2STR(&ip_info.ip));
}

i2s_chan_handle_t rx_handle = NULL;

void i2s_init() {
  // Use "chan_cfg" defined in constants/i2s_config_t.h
  ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx_handle));

  // Use "std_cfg" defined in constants/i2s_config_t.h
  ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_handle, &std_cfg));
  ESP_ERROR_CHECK(i2s_channel_enable(rx_handle));
}

extern "C" void app_main(void) {
  esp_err_t ret = nvs_flash_init();
  if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
      ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    ESP_ERROR_CHECK(nvs_flash_erase());
    ret = nvs_flash_init();
  }
  ESP_ERROR_CHECK(ret);

  ESP_LOGI(TAG, "Starting Audio Streamer (FIXED VERSION)...");

  // 1. Connect to Wi-Fi
  wifi_init_sta();

  // 2. Setup UDP Socket
  int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
  if (sock < 0) {
    ESP_LOGE(TAG, "Unable to create socket: errno %d", errno);
    return;
  }

  // FIX 3: Tăng kích thước bộ đệm gửi (TX Buffer) lên 32KB
  // Đây là thuốc chữa lỗi "TX Err 12"
  int snd_buf_size = 32 * 1024;
  int err_opt = setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &snd_buf_size, sizeof(snd_buf_size));
  if (err_opt != 0) ESP_LOGE(TAG, "Failed to set buffer size");

  struct sockaddr_in dest_addr;
  dest_addr.sin_addr.s_addr = inet_addr(PC_IP_ADDR);
  dest_addr.sin_family = AF_INET;
  dest_addr.sin_port = htons(PC_PORT);

  // 3. Init I2S
  i2s_init();

  ESP_LOGI(TAG, "Streaming to %s:%d", PC_IP_ADDR, PC_PORT);

  // 4. Loop
  size_t bytes_read = 0;
  
  size_t buffer32_len = FRAME_SIZE * sizeof(int32_t); 
  int32_t *buffer32 = (int32_t *)malloc(buffer32_len);
  int16_t *buffer16 = (int16_t *)malloc(FRAME_SIZE * sizeof(int16_t));

  if (!buffer32 || !buffer16) {
      ESP_LOGE(TAG, "Failed to allocate memory");
      return;
  }

  while (1) {
    // Read exactly one frame
    if (i2s_channel_read(rx_handle, buffer32, buffer32_len, &bytes_read, 1000) == ESP_OK) {
      
      int samples_read = bytes_read / 4; 
      
      // Convert 32-bit -> 16-bit (MONO extraction from STEREO stream)
      // I2S is reading STEREO (L, R, L, R, ...) but only LEFT channel has data (L/R pin grounded)
      // Extract only LEFT channel (every other sample)
      int samples_mono = samples_read / 2; // Half of stereo samples
      for (int i = 0; i < samples_mono; i++) {
        // Take only LEFT channel (index 0, 2, 4, 6, ...)
        // Using >> 16 for lower sensitivity (reduces quantization noise on quiet sounds)
        buffer16[i] = (int16_t)(buffer32[i * 2] >> 16);
      }
      
      // DEBUG: Print BOTH channels to see where data actually is
      static int log_counter = 0;
      if (log_counter++ % 20 == 0 && samples_read >= 6) {
        int16_t L0 = (int16_t)(buffer32[0] >> 16);  // LEFT channel sample 0
        int16_t R0 = (int16_t)(buffer32[1] >> 16);  // RIGHT channel sample 0
        int16_t L1 = (int16_t)(buffer32[2] >> 16);  // LEFT channel sample 1
        int16_t R1 = (int16_t)(buffer32[3] >> 16);  // RIGHT channel sample 1
        int16_t L2 = (int16_t)(buffer32[4] >> 16);  // LEFT channel sample 2
        int16_t R2 = (int16_t)(buffer32[5] >> 16);  // RIGHT channel sample 2
        
        ESP_LOGI(TAG, "L:[%d,%d,%d] R:[%d,%d,%d] | RAW32_L0=%ld RAW32_R0=%ld", 
                 L0, L1, L2, R0, R1, R2, 
                 (long)buffer32[0], (long)buffer32[1]);
      }

      // Send 16-bit UDP packet (MONO data)
      int err = sendto(sock, buffer16, samples_mono * 2, 0, (struct sockaddr *)&dest_addr,
                 sizeof(dest_addr));
      
      if (err < 0) {
        // FIX 5: Xử lý lỗi tràn bộ nhớ (Err 12)
        if (errno == 12) {
             // Nghỉ 10ms để giải phóng bộ đệm Wifi rồi mới gửi tiếp
             vTaskDelay(pdMS_TO_TICKS(10));
        } else {
             ESP_LOGE(TAG, "TX Err %d", errno);
        }
      }
    } else {
       vTaskDelay(1);
    }
  }
}