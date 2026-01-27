#include "driver/i2s_std.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_system.h"
#include "esp_timer.h"
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
// RNNoise includes
extern "C"
{
#include "rnnoise.h"
}

// =============================================================
// CONFIGURATION
// =============================================================
#define WIFI_SSID "J19"
#define WIFI_PASS "hoangchimbe"
#define PC_IP_ADDR "192.168.1.12" // Hãy đổi thành IP máy tính hiện tại
#define PC_PORT 12345

static const char *TAG = "AUDIO";
static EventGroupHandle_t s_wifi_event_group;
#define WIFI_CONNECTED_BIT BIT0

static void event_handler(void *arg, esp_event_base_t event_base,
                          int32_t event_id, void *event_data)
{
  if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START)
  {
    esp_wifi_connect();
  }
  else if (event_base == WIFI_EVENT &&
           event_id == WIFI_EVENT_STA_DISCONNECTED)
  {
    esp_wifi_connect();
    ESP_LOGI(TAG, "Retrying to connect to the AP");
  }
  else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP)
  {
    ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
    ESP_LOGI(TAG, "Got IP:" IPSTR, IP2STR(&event->ip_info.ip));
    xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
  }
}

void wifi_init_sta(void)
{
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

#include "driver/gpio.h" // Add GPIO header

i2s_chan_handle_t rx_handle = NULL;

// === RNNoise Global State ===
static DenoiseState *rnn_state = NULL;
static volatile bool process_rnnoise = true; // Default ON, volatile for task visibility
static float input_buffer[480];  // Float buffer for RNNoise input
static float output_buffer[480]; // Float buffer for RNNoise output

// === RNNoise Helper Functions ===

// Convert int16 samples to float for RNNoise input
void convert_int16_to_float(const int16_t *in, float *out, size_t len)
{
  for (size_t i = 0; i < len; i++)
  {
    out[i] = (float)in[i]; // Range: -32768.0 to 32767.0
  }
}

// Convert float samples back to int16 for UDP transmission
void convert_float_to_int16(const float *in, int16_t *out, size_t len)
{
  for (size_t i = 0; i < len; i++)
  {
    float val = in[i];
    // Clamp to int16 range
    if (val > 32767.0f)
      val = 32767.0f;
    if (val < -32768.0f)
      val = -32768.0f;
    out[i] = (int16_t)val;
  }
}

void i2s_init()
{
  // Use "chan_cfg" defined in constants/i2s_config_t.h
  ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx_handle));

  // Use "std_cfg" defined in constants/i2s_config_t.h
  ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_handle, &std_cfg));
  ESP_ERROR_CHECK(i2s_channel_enable(rx_handle));
}

// Task handle for audio streaming
TaskHandle_t s_audio_task_handle = NULL;

void udp_audio_task(void *pvParameters)
{
  // 1. Setup UDP Socket
  int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
  if (sock < 0)
  {
    ESP_LOGE(TAG, "Unable to create socket: errno %d", errno);
    vTaskDelete(NULL);
    return;
  }

  // Set timeout prevent blocking forever
  struct timeval timeout;
  timeout.tv_sec = 0;
  timeout.tv_usec = 100000; // 100ms
  setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof timeout);

  // Increase Socket TX Buffer (optional, ignore error if fails)
  int snd_buf_size = 32 * 1024;
  setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &snd_buf_size, sizeof(snd_buf_size));

  struct sockaddr_in dest_addr;
  dest_addr.sin_addr.s_addr = inet_addr(PC_IP_ADDR);
  dest_addr.sin_family = AF_INET;
  dest_addr.sin_port = htons(PC_PORT);

  // 2. Init I2S
  // Note: I2S init moved here or kep in main? Better in main or before loop.
  // Actually, keeping i2s_init() global is fine.

  ESP_LOGI(TAG, "Streaming to %s:%d", PC_IP_ADDR, PC_PORT);

  size_t bytes_read = 0;
  size_t buffer32_len = FRAME_SIZE * 2 * sizeof(int32_t); // x2 for Stereo (L+R)

  // Allocate buffers
  int32_t *buffer32 = (int32_t *)malloc(buffer32_len);
  int16_t *buffer16 = (int16_t *)malloc(FRAME_SIZE * sizeof(int16_t));

  if (!buffer32 || !buffer16)
  {
    ESP_LOGE(TAG, "Failed to allocate audio buffers");
    close(sock);
    vTaskDelete(NULL);
    return;
  }

  // 5. Config Button (Boot Button = GPIO 0)
  gpio_config_t io_conf = {};
  io_conf.intr_type = GPIO_INTR_DISABLE;
  io_conf.mode = GPIO_MODE_INPUT;
  io_conf.pin_bit_mask = (1ULL << GPIO_NUM_0);
  io_conf.pull_down_en = GPIO_PULLDOWN_DISABLE;
  io_conf.pull_up_en = GPIO_PULLUP_ENABLE;
  gpio_config(&io_conf);

  int last_btn_state = 1;

  while (1)
  {
    // === Button Logic (Simple Polling) ===
    int btn_state = gpio_get_level(GPIO_NUM_0);
    if (last_btn_state == 1 && btn_state == 0) {
        // Falling edge (Pressed)
        process_rnnoise = !process_rnnoise;
        ESP_LOGW(TAG, "Noise Reduction Toggled: %s", process_rnnoise ? "ON" : "OFF");
        vTaskDelay(pdMS_TO_TICKS(200)); // Debounce
    }
    last_btn_state = btn_state;

    // Read exactly one frame
    // We expect 480 samples. 32-bit mono = 480 * 4 bytes = 1920 bytes.
    if (i2s_channel_read(rx_handle, buffer32, buffer32_len, &bytes_read, 1000) == ESP_OK)
    {


      int samples_read = bytes_read / 4;

      // FIX: Revert to STEREO handling for INMP441
      // I2S is reading STEREO (L, R, L, R, ...) but only LEFT channel has valid data
      int samples_mono = samples_read / 2;

      for (int i = 0; i < samples_mono; i++)
      {
        // Take only LEFT channel (index 0, 2, 4...)
        // buffer32[i * 2] is Left channel
        buffer16[i] = (int16_t)(buffer32[i * 2] >> 16); 
      }

      // === RNNoise Processing (ENABLED) ===
      // #if 0
      if (process_rnnoise && samples_mono == 480)
      {
        // Step 1: Convert int16 → float
        convert_int16_to_float(buffer16, input_buffer, samples_mono);

        // Step 2: Apply noise suppression
        uint64_t rnn_start = esp_timer_get_time();
        rnnoise_process_frame(rnn_state, output_buffer, input_buffer);
        uint64_t rnn_elapsed = esp_timer_get_time() - rnn_start;

        // Step 3: Convert float → int16
        convert_float_to_int16(output_buffer, buffer16, samples_mono);

        // Log performance periodically
        static int perf_counter = 0;
        if (perf_counter++ % 100 == 0)
        {
          ESP_LOGI(TAG, "RNNoise: %llu us/frame, Heap: %d",
                   rnn_elapsed, esp_get_free_heap_size());
        }
      }
      else if (samples_mono != 480)
      {
        // Only log warning occasionally to avoid serial spam causing lags
        static int warn_counter = 0;
        if (warn_counter++ % 100 == 0)
        {
          ESP_LOGW(TAG, "Frame size mismatch: %d (expected 480)", samples_mono);
        }
      }
      // #endif

      // Send 16-bit UDP packet
      int err = sendto(sock, buffer16, samples_mono * 2, 0, (struct sockaddr *)&dest_addr,
                       sizeof(dest_addr));

      if (err < 0)
      {
        if (errno == 12)
        { // ENOMEM
          vTaskDelay(pdMS_TO_TICKS(10));
        }
        else
        {
          ESP_LOGE(TAG, "Tx Err: %d", errno);
        }
      }

      // FIX: Yield to IDLE task to reset Watchdog (WDT)
      // Since RNNoise takes ~8-9ms, we have a little CPU time left.
      // vTaskDelay(1) ensures the IDLE task gets a chance to run.
      vTaskDelay(1);
    }
    else
    {
      // I2S Read timeout/fail
      vTaskDelay(1);
    }

    // Feed watchdog if task takes too long (though I2S read usually blocks)
    // No explicit vTaskDelay needed if I2S blocks, but good practice if loop is fast
  }

  free(buffer32);
  free(buffer16);
  close(sock);
  vTaskDelete(NULL);
}

extern "C" void app_main(void)
{
  esp_err_t ret = nvs_flash_init();
  if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
      ret == ESP_ERR_NVS_NEW_VERSION_FOUND)
  {
    ESP_ERROR_CHECK(nvs_flash_erase());
    ret = nvs_flash_init();
  }
  ESP_ERROR_CHECK(ret);

  ESP_LOGI(TAG, "Starting Audio Streamer (RNNoise + Task Version)...");

  // 1. Connect to Wi-Fi
  wifi_init_sta();

  // 2. Initialize RNNoise
  ESP_LOGI(TAG, "Initializing RNNoise...");
  rnn_state = rnnoise_create(NULL);
  if (!rnn_state)
  {
    ESP_LOGE(TAG, "Failed to create RNNoise state!");
    process_rnnoise = false;
  }
  else
  {
    ESP_LOGI(TAG, "RNNoise initialized. Heap: %d", esp_get_free_heap_size());
    process_rnnoise = true;
  }

  // 3. Init I2S
  i2s_init();

  // 4. Create Audio Task
  // Increased Stack to 128KB (was 64KB) to accommodate RNNoise + WiFi
  xTaskCreatePinnedToCore(udp_audio_task, "audio_task", 131072, NULL, 5, &s_audio_task_handle, 1);
}