import numpy as np
import matplotlib.pyplot as plt

def generate_anatomy_plot(noise_type="stationary"):
    # ==========================================
    # GIAI ĐOẠN 1: MÁP XẠ KHUNG DỮ LIỆU C GỐC VÀO PYTHON
    # ==========================================
    np.random.seed(42 if noise_type=="stationary" else 15)

    # MÁP XẠ 1: Kích thước Khung và Phổ (File: denoise.h)
    N = 480  
    freq_bins = 481

    # MÁP XẠ 2: Khối thu Tín hiệu Thô (File: denoise.c)
    t = np.linspace(0, 0.01, N)
    clean_wave = np.sin(2 * np.pi * 400 * t) * 5000  # Giọng người lý tưởng
    
    background_noise = np.random.normal(0, 1500, N) # Nhiễu nền tĩnh (Quạt máy/Điều hòa)
    if noise_type == "non_stationary":
        # Tạo Nhiễu không tĩnh (Ví dụ: Tiếng gõ phím cạch cạch đột ngột)
        transient_spike = np.zeros(N)
        transient_spike[150:190] = np.random.normal(0, 15000, 40) * np.hanning(40) # Sốc cực mạnh!
        noise_wave = background_noise + transient_spike
        title_suffix = "(Lẫn cú sốc Nhiễu Không Tĩnh)"
    else:
        noise_wave = background_noise
        title_suffix = "(Chỉ có Nhiễu Tĩnh đều đặn)"
        
    raw_pcm = clean_wave + noise_wave

    # MÁP XẠ 3: Phân tích FFT (File: denoise.c)
    raw_fft = np.abs(np.fft.rfft(raw_pcm * np.hanning(N), n=960)) 
    raw_db = 20 * np.log10(raw_fft + 1e-6)

    # MÁP XẠ 4: Dò tìm Pitch (Hàm Tương quan) (File: pitch.c / denoise.h)
    PITCH_MIN_PERIOD = 60
    PITCH_MAX_PERIOD = 768
    T0 = 120 # Giả lập giọng nói có chu kỳ T=120 mẫu (Ngay 400Hz)

    pitch_lags = np.arange(PITCH_MIN_PERIOD, PITCH_MAX_PERIOD)
    pitch_corr = np.random.normal(0, 0.15, len(pitch_lags))
    pitch_corr[T0 - PITCH_MIN_PERIOD] = 0.95  # Đỉnh chính do find_best_pitch() quét được
    pitch_corr[(T0*2) - PITCH_MIN_PERIOD] = 0.45  # Đỉnh ảo bị rnn_remove_doubling chặt đi

    # MÁP XẠ 5: Đặc trưng Tần số Tai người (Bark Energy) (File: denoise.c)
    NB_BANDS = 32
    bark_energy = np.random.uniform(20, 80, NB_BANDS)
    bark_energy[3:8] += 40 # Năng lượng giọng nói tập trung
    bark_energy[12:15] += 20
    if noise_type == "non_stationary":
        bark_energy += np.random.uniform(10, 50, NB_BANDS) # Cú sốc transient bùng lên mọi dải tần

    # MÁP XẠ 6: Mạng Neural Network Dập Nhiễu (File: rnn.c / denoise.c)
    ai_gains = np.random.uniform(0.0, 0.2, NB_BANDS)
    ai_gains[3:8] = 0.95 # Mệnh lệnh cho lọc Pitch: THẢ LỎNG (Giữ tiếng người)
    ai_gains[12:15] = 0.85 # Mệnh lệnh cho lọc Pitch: THẢ LỎNG (Giữ họa âm)
    if noise_type == "non_stationary":
        ai_gains[ai_gains < 0.8] = np.random.uniform(0.0, 0.05, len(ai_gains[ai_gains < 0.8])) # Cắt gắt hơn

    # MÁP XẠ 7: Nội suy 32 Mặt nạ lên 481 Dải phổ (File: denoise.c)
    interp_mask = np.interp(np.linspace(0, 31, freq_bins), np.arange(32), ai_gains)

    # MÁP XẠ 8: Trừ Nhiễu Phổ (Spectral Subtraction) (File: denoise.c)
    clean_fft = raw_fft * interp_mask
    clean_db = 20 * np.log10(clean_fft + 1e-6)

    # MÁP XẠ 9: Tổng hợp Lại Thành PCM Mượt (File: denoise.c)
    clean_pcm_out = np.fft.irfft(clean_fft * np.exp(1j * np.random.uniform(-np.pi, np.pi, freq_bins)))
    clean_pcm_out = clean_pcm_out[:N] * max(np.abs(raw_pcm))/max(np.abs(clean_pcm_out)) # Chuẩn hóa

    # ==========================================
    # GIAI ĐOẠN 2: VẼ THÀNH ẢNH (8 TRỤC)
    # ==========================================
    plt.style.use('default') 
    fig, axs = plt.subplots(4, 2, figsize=(15, 13))
    
    header_title = f"Phẫu thuật Dữ liệu RNNoise - Phân tích loại {('NHIỄU KHÔNG TĨNH (Sốc)' if noise_type == 'non_stationary' else 'NHIỄU TĨNH (Nền)')}"
    fig.suptitle(header_title, fontsize=18, fontweight='bold', color=('#D32F2F' if noise_type == 'non_stationary' else '#1565C0'))
    plt.subplots_adjust(hspace=0.4, wspace=0.15)

    # [0, 0]: Raw PCM
    axs[0, 0].plot(t, raw_pcm, color='#D32F2F', linewidth=1.5)
    axs[0, 0].set_title(f"1. Tín hiệu đầu vào {title_suffix}", fontweight='bold')
    axs[0, 0].set_xlabel("Thời gian (s)")
    axs[0, 0].grid(True, linestyle='--', alpha=0.5)

    # [0, 1]: FFT Spectrum
    axs[0, 1].plot(raw_db, color='#546E7A', linewidth=1.5)
    axs[0, 1].set_title("2. Phổ tần số (FFT 481 Bins)", fontweight='bold')
    axs[0, 1].set_ylabel("Biên độ (dB)")
    axs[0, 1].grid(True, linestyle='--', alpha=0.5)

    # [1, 0]: Pitch Correlation
    axs[1, 0].plot(pitch_lags, pitch_corr, color='#8E24AA', linewidth=1.5)
    axs[1, 0].axvline(x=T0, color='r', linestyle='-', alpha=0.8)
    axs[1, 0].set_title(f"3. Dò Cao độ (Pitch Correlation từ Lag {PITCH_MIN_PERIOD} -> {PITCH_MAX_PERIOD})", fontweight='bold')
    axs[1, 0].text(T0 + 20, 0.8, f"Pitch T0={T0}", color='r', fontweight='bold')
    axs[1, 0].text((T0*2) + 15, 0.45, "Đỉnh giả", color='navy', fontweight='bold')
    axs[1, 0].set_xlim(50, 800)
    axs[1, 0].grid(True, linestyle='--', alpha=0.5)

    # [1, 1]: Bark Energy
    axs[1, 1].bar(np.arange(32), bark_energy, color='#0288D1', edgecolor='black')
    axs[1, 1].set_title("4. Đặc trưng Phổ thính giác (32 Bark Bands)", fontweight='bold')
    axs[1, 1].set_xticks(np.arange(0, 32, 5))
    axs[1, 1].grid(axis='y', linestyle='--', alpha=0.5)

    # [2, 0]: AI Model Gains
    axs[2, 0].bar(np.arange(32), ai_gains, color='#FF8F00', edgecolor='black')
    axs[2, 0].set_title("5. Trọng tài AI Phán quyết (32 Gains Vector)", fontweight='bold')
    axs[2, 0].set_ylim(0, 1.1)
    axs[2, 0].axhline(y=0.5, color='gray', linestyle='--')
    axs[2, 0].grid(axis='y', linestyle='--', alpha=0.5)

    # [2, 1]: Mask Interpolation
    axs[2, 1].plot(interp_mask, color='#F57C00', linewidth=2.5)
    axs[2, 1].fill_between(range(481), interp_mask, color='#FFE0B2', alpha=0.6)
    axs[2, 1].set_title("6. Nội suy Mặt nạ chặn nhiễu (32 -> 481 Bins)", fontweight='bold')
    axs[2, 1].grid(True, linestyle='--', alpha=0.5)

    # [3, 0]: Spectral Subtraction
    axs[3, 0].plot(raw_db, color='#B0BEC5', alpha=0.8, label="Phổ trước (Có nhiễu)")
    axs[3, 0].plot(clean_db, color='#2E7D32', linewidth=1.5, label="Phổ sau (Đã lọc)")
    axs[3, 0].set_title("7. Áp dụng Mặt nạ Lọc lên Phổ FFT", fontweight='bold')
    axs[3, 0].legend(loc="upper right")
    axs[3, 0].grid(True, linestyle='--', alpha=0.5)

    # [3, 1]: Clean PCM
    axs[3, 1].plot(t, clean_wave, color='#81C784', linestyle='dashed', alpha=0.7, label="Sóng hài cơ bản (Lý tưởng)")
    axs[3, 1].plot(t, clean_pcm_out, color='#2E7D32', linewidth=1.5, label="Giọng sau khi lọc")
    axs[3, 1].set_title("8. Tín hiệu đầu ra sau IFFT (Sạch nhiễu)", fontweight='bold')
    axs[3, 1].set_xlabel("Thời gian (s)")
    axs[3, 1].legend(loc="upper right")
    axs[3, 1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    filename = f"rnnoise_anatomy_{noise_type}.png"
    plt.savefig(filename, dpi=300)
    print(f"Đã xuất file Slide ảnh: {filename}")
    plt.close()

if __name__ == "__main__":
    generate_anatomy_plot("stationary")
    generate_anatomy_plot("non_stationary")
