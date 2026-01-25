#include "driver/i2s_std.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <math.h>
#include <stdio.h>

// User wiring
#define I2S_BCLK GPIO_NUM_5
#define I2S_WS GPIO_NUM_6
#define I2S_DOUT GPIO_NUM_7

#define SAMPLE_RATE 48000
#define FRAME 480

static i2s_chan_handle_t tx;
static const char *TAG = "TONE";

extern "C" void app_main() {
  printf("Resilient 1kHz Tone Test (16-bit Mono)...\n");

  i2s_chan_config_t chan_cfg =
      I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
  chan_cfg.auto_clear = true;
  ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, &tx, NULL));

  i2s_std_config_t cfg = {
      .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
      .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT,
                                                      I2S_SLOT_MODE_MONO),
      .gpio_cfg =
          {
              .mclk = I2S_GPIO_UNUSED,
              .bclk = I2S_BCLK,
              .ws = I2S_WS,
              .dout = I2S_DOUT,
              .din = I2S_GPIO_UNUSED,
          },
  };

  ESP_ERROR_CHECK(i2s_channel_init_std_mode(tx, &cfg));
  ESP_ERROR_CHECK(i2s_channel_enable(tx));

  // Pre-calculate buffer to save CPU
  int16_t buf[FRAME];
  float phase = 0;
  float step = 2.0f * (float)M_PI * 1000.0f / SAMPLE_RATE;

  for (int i = 0; i < FRAME; i++) {
    buf[i] = (int16_t)(sinf(phase) * 12000.0f);
    phase += step;
  }

  size_t w = 0;
  int count = 0;

  while (1) {
    esp_err_t ret = i2s_channel_write(tx, buf, sizeof(buf), &w, portMAX_DELAY);

    if (ret != ESP_OK) {
      ESP_LOGE(TAG, "Write Failed! %s", esp_err_to_name(ret));
      vTaskDelay(pdMS_TO_TICKS(100)); // Wait a bit if error
    } else {
      // Normal operation
      if (count++ % 100 == 0) { // Log occasionally
        printf(".");            // Heartbeat
        fflush(stdout);
      }
    }

    // Safety yield (Critical for WDT fix)
    vTaskDelay(1);
  }
}
