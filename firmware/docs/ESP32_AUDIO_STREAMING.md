# ESP32 Audio Streaming System

## Overview

Real-time audio streaming from INMP441 I2S microphone to PC via WiFi UDP.

## Hardware Configuration

### ESP32-S3 Super Mini + INMP441 Microphone

| INMP441 Pin | ESP32 GPIO | Function                              |
| ----------- | ---------- | ------------------------------------- |
| VDD         | 3.3V       | Power (⚠️ NOT 5V)                     |
| GND         | GND        | Ground                                |
| SD          | GPIO 4     | I2S Data Input                        |
| SCK         | GPIO 5     | I2S Bit Clock                         |
| WS          | GPIO 6     | I2S Word Select                       |
| L/R         | GND        | Channel Select (Left=GND, Right=3.3V) |

> ⚠️ **IMPORTANT**: L/R pin MUST be connected to GND. Connecting to 3.3V may cause short circuit on some INMP441 modules.

## Audio Specifications

- **Sample Rate**: 48,000 Hz (48kHz)
- **Bit Depth**: 16-bit PCM (read as 32-bit, converted to 16-bit)
- **Channels**: Mono (extracted from stereo I2S stream)
- **Frame Size**: 480 samples (10ms per frame)
- **I2S Mode**: Philips Standard (compatible with INMP441)

## Network Configuration

### WiFi Settings

Edit in `src/main.cpp`:

```cpp
#define WIFI_SSID "Your_SSID"
#define WIFI_PASS "Your_Password"
#define PC_IP_ADDR "192.168.x.x"  // Your PC IP address
#define PC_PORT 12345
```

### Features

- UDP streaming protocol
- WiFi Power Save disabled for consistent throughput
- 32KB TX buffer to prevent packet loss
- Automatic reconnection on disconnect

## Project Structure

```
firmware/esp32-rnnoise/
├── src/
│   ├── main.cpp                    # Main application logic
│   └── constants/
│       └── i2s_config_t.h         # I2S & GPIO configuration
├── platformio.ini                  # Build configuration
└── sdkconfig.defaults             # ESP-IDF settings
```

## Key Implementation Details

### Stereo to Mono Extraction

I2S peripheral reads stereo data (L, R, L, R, ...) but only LEFT channel contains mic data:

```cpp
int samples_mono = samples_read / 2;
for (int i = 0; i < samples_mono; i++) {
    buffer16[i] = (int16_t)(buffer32[i * 2] >> 16);
}
```

### 32-bit to 16-bit Conversion

INMP441 outputs 24-bit data in 32-bit frames. Shift right by 16 to extract top 16 bits:

```cpp
buffer16[i] = (int16_t)(buffer32[i] >> 16);
```

## PC Receiver (Python)

### Installation

```bash
pip install pyaudio
```

### Run

```bash
python scripts/udp_player.py
```

### Features

- Jitter buffer (2 frames = 20ms latency)
- Automatic packet dropout handling
- Real-time playback via PyAudio

## Build & Flash

### Using PlatformIO

```bash
cd firmware/esp32-rnnoise

# Build
pio run

# Upload (replace COM3 with your port)
pio run -t upload --upload-port COM3

# Monitor serial output
pio run -t monitor --upload-port COM3

# Or combined
pio run -t upload -t monitor --upload-port COM3
```

### Find ESP32 Port

```bash
pio device list
```

## Troubleshooting

### No audio / All zeros

- Check GPIO pin connections
- Verify VDD is 3.3V (NOT 5V)
- Ensure L/R pin is connected to GND
- Check ESP32 port in Device Manager (Windows)

### Clicking/Popping sounds

- Network jitter - increase jitter buffer size in `udp_player.py`
- WiFi interference - move closer to router

### "TX Err 12" in serial monitor

- Network buffer full
- Already fixed by disabling WiFi power save and increasing TX buffer

### Loud sounds distorted

- Clipping due to incorrect bit shift
- Verify using `>> 16` in conversion code

## Performance

- **Latency**: ~30-50ms (20ms jitter buffer + network delay)
- **CPU Usage**: ~15% on ESP32-S3 @ 240MHz
- **Network Bandwidth**: ~1.5 Mbps (48kHz × 16-bit × mono)
- **Packet Rate**: ~100 packets/second (480 samples/packet)

## Next Steps

This audio streaming system provides clean, real-time audio data ready for:

1. **RNNoise Integration**: Real-time noise suppression
2. **VAD (Voice Activity Detection)**: Optimize bandwidth
3. **Audio Recording**: Save to WAV file on PC
4. **Speech Recognition**: Feed to ASR models

## References

- [INMP441 Datasheet](https://invensense.tdk.com/products/digital/inmp441/)
- [ESP-IDF I2S Driver](https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/api-reference/peripherals/i2s.html)
- [RNNoise Project](https://gitlab.xiph.org/xiph/rnnoise)
