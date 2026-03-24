import numpy as np

def print_data_mockup():
    print("="*60)
    print(" 🛠️  MÔ PHỎNG DỮ LIỆU CÁC PROBE TRONG RNNOISE  🛠️")
    print("="*60)

    # 1. P0 & P6: Raw/Clean PCM (10ms @ 48kHz = 480 samples)
    # Value range: -32768 to 32767
    pcm_data = np.random.randint(-5000, 5000, size=480, dtype=np.int16)
    print("\n[P0: Raw PCM] Khay chứa 480 dao động âm thanh (int16)")
    print(f"Shape: {pcm_data.shape} | Type: {pcm_data.dtype.name}")
    print(f"Data snippet: [{pcm_data[0]}, {pcm_data[1]}, {pcm_data[2]}, ..., {pcm_data[-2]}, {pcm_data[-1]}]")

    # 2. P1: FFT Spect (481 bins for 48kHz)
    # Value range: typically 0.0 to highly positive
    fft_data = np.abs(np.random.randn(481)).astype(np.float32) * 10.0
    print("\n[P1: FFT Spect] Khay chứa 481 mức năng lượng tần số (float32)")
    print(f"Shape: {fft_data.shape} | Type: {fft_data.dtype.name}")
    print(f"Data snippet: [{fft_data[0]:.2f}, {fft_data[1]:.2f}, {fft_data[2]:.2f}, ..., {fft_data[-2]:.2f}, {fft_data[-1]:.2f}]")

    # 3. P2: Pitch Lag (143 correlation points)
    # Represents correlation at different time lags
    pitch_data = np.random.uniform(-1.0, 1.0, size=143).astype(np.float32)
    print("\n[P2: Pitch Lag] Khay chứa 143 hệ số tương quan tìm giọng nói (float32)")
    print(f"Shape: {pitch_data.shape} | Type: {pitch_data.dtype.name}")
    print(f"Data snippet: [{pitch_data[0]:.3f}, {pitch_data[1]:.3f}, {pitch_data[2]:.3f}, ..., {pitch_data[-2]:.3f}, {pitch_data[-1]:.3f}]")

    # 4. P3: Bark Energy (32 frequency bands tailored to human hearing)
    # Value range: 0.0 to ~100.0+ 
    bark_data = np.random.uniform(0.0, 50.0, size=32).astype(np.float32)
    print("\n[P3: Bark Energy] Khay chứa 32 cụm năng lượng thính giác (float32)")
    print(f"Shape: {bark_data.shape} | Type: {bark_data.dtype.name}")
    print(f"Data snippet (all 32 bands):\n {np.round(bark_data[:16], 1)}\n {np.round(bark_data[16:], 1)}")

    # 5. P4: AI Gains (32 suppression factors decided by GRU)
    # Value range: 0.0 (mute) to 1.0 (pass through)
    gains_data = np.random.uniform(0.0, 1.0, size=32).astype(np.float32)
    print("\n[P4: AI Gains] Khay chứa 32 hệ số AI quyết định cắt nhiễu (float32)")
    print(f"Shape: {gains_data.shape} | Type: {gains_data.dtype.name}")
    print(f"Data snippet (all 32 bands):\n {np.round(gains_data[:16], 2)}\n {np.round(gains_data[16:], 2)}")

    # 6. P5: VAD (Voice Activity Detection probability)
    # Scalar value 0.0 to 1.0
    vad_data = np.float32(0.875)
    print(f"\n[P5: VAD] Xác suất có tiếng người (float32)")
    print(f"Value: {vad_data:.3f} (Tức là AI tự tin 87.5% frame rày là có giọng nói)")
    print("="*60)

if __name__ == "__main__":
    print_data_mockup()
