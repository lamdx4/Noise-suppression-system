#include "driver/i2s_std.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "esp_heap_caps.h" // Thêm header cho heap_caps_malloc
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
#include "freertos/ringbuf.h" // Thêm thư viện RingBuffer
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
#define PC_IP_ADDR "192.168.1.12"
#define PC_PORT 12345

#define BUFFER_SIZE (FRAME_SIZE * sizeof(int16_t)) // 960 bytes
#define RING_BUF_SIZE (20 * 1024)                  // 20KB bồn chứa (khoảng 20 frames)

static const char *TAG = "AUDIO";
static EventGroupHandle_t s_wifi_event_group;
#define WIFI_CONNECTED_BIT BIT0

RingbufHandle_t audio_ring_buf = NULL;
i2s_chan_handle_t rx_handle = NULL;

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
    }
    else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP)
    {
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

    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &event_handler, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &event_handler, NULL, NULL));

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
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    esp_wifi_set_max_tx_power(52);

    xEventGroupWaitBits(s_wifi_event_group, WIFI_CONNECTED_BIT, pdFALSE, pdFALSE, portMAX_DELAY);
}

void i2s_init()
{
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx_handle));
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_handle, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(rx_handle));
}

// =============================================================
// CORE 1: Sampler Task (Chuyên trách đọc Mic & Convert)
// =============================================================
void i2s_sampler_task(void *pvParameters)
{
    size_t bytes_read = 0;
    size_t buffer32_len = FRAME_SIZE * sizeof(int32_t);
    int32_t *buffer32 = (int32_t *)heap_caps_malloc(buffer32_len, MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL);
    int16_t *buffer16 = (int16_t *)heap_caps_malloc(FRAME_SIZE * sizeof(int16_t), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);

    if (!buffer32 || !buffer16)
    {
        ESP_LOGE(TAG, "Sampler Task: Failed to allocate memory in Internal RAM");
        vTaskDelete(NULL);
    }

    ESP_LOGI(TAG, "Sampler Task started on Core 1");

    while (1)
    {
        if (i2s_channel_read(rx_handle, buffer32, buffer32_len, &bytes_read, portMAX_DELAY) == ESP_OK)
        {
            int samples_mono = bytes_read / 4; // Vì dùng I2S_SLOT_MODE_MONO, mỗi sample 32-bit = 4 bytes

            // Convert to 16-bit Mono
            int32_t *src = buffer32;
            int16_t *dst = buffer16;
            for (int i = 0; i < samples_mono; i++)
            {
                *dst++ = (int16_t)((*src) >> 16);
                src += 2;
            }

            // Đẩy vào RingBuffer
            if (xRingbufferSend(audio_ring_buf, buffer16, samples_mono * 2, pdMS_TO_TICKS(10)) != pdTRUE)
            {
                // Buffer đầy - có thể do Wifi chậm. Bỏ qua frame này để giữ tính thời gian thực
                // ESP_LOGW(TAG, "RingBuffer Full!");
            }
        }
    }
}

// =============================================================
// CORE 0: UDP Sender Task (Chuyên trách truyền tải dữ liệu)
// =============================================================
void udp_sender_task(void *pvParameters)
{
    struct sockaddr_in dest_addr;
    dest_addr.sin_addr.s_addr = inet_addr(PC_IP_ADDR);
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(PC_PORT);

    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (sock < 0)
    {
        ESP_LOGE(TAG, "UDP Sender Task: Unable to create socket: errno %d", errno);
        vTaskDelete(NULL);
    }

    int snd_buf_size = 32 * 1024;
    int err_opt = setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &snd_buf_size, sizeof(snd_buf_size));
    if (err_opt != 0)
        ESP_LOGE(TAG, "UDP Sender Task: Failed to set buffer size");

    ESP_LOGI(TAG, "UDP Sender Task started on Core 0. Streaming to %s:%d", PC_IP_ADDR, PC_PORT);

    while (1)
    {
        size_t item_size;
        // Lấy dữ liệu từ RingBuffer
        void *item = xRingbufferReceive(audio_ring_buf, &item_size, portMAX_DELAY);

        if (item != NULL)
        {
            int err = sendto(sock, item, item_size, 0, (struct sockaddr *)&dest_addr, sizeof(dest_addr));

            if (err < 0)
            {
                if (errno == 12)
                {
                    vTaskDelay(pdMS_TO_TICKS(2)); // Đợi nhẹ để Wifi stack thở
                }
                else
                {
                    ESP_LOGE(TAG, "UDP Sender Task: TX Err %d, check Server connection", errno);
                    vTaskDelay(pdMS_TO_TICKS(100));
                }
            }

            // MÀN QUAN TRỌNG: Trả lại bộ nhớ sau khi dùng xong
            vRingbufferReturnItem(audio_ring_buf, item);
        }
    }
}

extern "C" void app_main(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND)
    {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    // 1. Khởi tạo RingBuffer
    // NOSPLIT đảm bảo mỗi packet 960 bytes được giữ nguyên khối
    audio_ring_buf = xRingbufferCreate(RING_BUF_SIZE, RINGBUF_TYPE_NOSPLIT);
    if (audio_ring_buf == NULL)
    {
        ESP_LOGE(TAG, "Failed to create RingBuffer");
        return;
    }

    // 2. Khởi tạo Networking & I2S
    wifi_init_sta();
    i2s_init(); // 3. Tạo 2 Task trên 2 nhân khác nhau
    // Sampler: Core 1, Ưu tiên cao (5)
    xTaskCreatePinnedToCore(i2s_sampler_task, "Sampler", 4096, NULL, 5, NULL, 1);

    // Sender: Core 0 (cùng CPU với Wifi stack), Ưu tiên trung bình (4)
    xTaskCreatePinnedToCore(udp_sender_task, "Sender", 4096, NULL, 4, NULL, 0);

    ESP_LOGI(TAG, "Multi-core Engine deployed. Streaming to %s", PC_IP_ADDR);
}