#pragma once
#include "driver/i2s_std.h"
#include "hal/gpio_types.h"

// Golden Parameters for RNNoise
#define SAMPLE_RATE     48000
#define FRAME_SIZE      480   // 10ms Frame
#define I2S_PORT        I2S_NUM_0

// Pins Configuration (ESP32-S3 Super Mini)
#define I2S_SCK_IO      GPIO_NUM_5   // Nối chung SCK Mic + BCLK Loa
#define I2S_WS_IO       GPIO_NUM_6   // Nối chung WS Mic + LRC Loa
#define I2S_DIN_IO      GPIO_NUM_4   // Nối SD của Mic (Thu vào) - FIXED: User connected to GPIO4
#define I2S_DOUT_IO     GPIO_NUM_8   // <--- BỔ SUNG: Nối DIN của Loa (Phát ra)

// 1. Cấu hình Tài nguyên kênh (Channel Config) - Quyết định độ trễ
// Cái này quyết định DMA Buffer có đúng 480 hay không
static i2s_chan_config_t chan_cfg = {
    .id = I2S_PORT,
    .role = I2S_ROLE_MASTER,
    .dma_desc_num = 6,          // Số lượng bộ đệm (ít quá dễ vỡ tiếng, 4-6 là đẹp)
    .dma_frame_num = FRAME_SIZE,// <--- QUAN TRỌNG: Khóa cứng 480 mẫu/frame
    .auto_clear = true,         // Tự xóa buffer nếu lỗi (tránh tiếng rẹt rẹt)
};

// 2. Cấu hình chuẩn giao tiếp (Standard Config)
static i2s_std_config_t std_cfg = {
    // Clock: 48kHz
    .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
    
    // Slot: Dùng chuẩn Philips cho INMP441, 32-bit width để hứng trọn 24-bit data
    .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
    
    // GPIO Mapping
    .gpio_cfg = {
        .mclk = I2S_GPIO_UNUSED,
        .bclk = I2S_SCK_IO,
        .ws   = I2S_WS_IO,
        .dout = I2S_DOUT_IO,    // Đã thêm chân loa
        .din  = I2S_DIN_IO,     // Chân mic
        .invert_flags = {
            .mclk_inv = false,
            .bclk_inv = false,
            .ws_inv   = false,
        },
    },
};