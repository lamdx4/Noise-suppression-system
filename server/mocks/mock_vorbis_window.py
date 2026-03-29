import numpy as np
import matplotlib.pyplot as plt

def generate_vorbis_plot():
    print("Mô phỏng Chống Rò rỉ Phổ bằng Cửa sổ Vorbis...")
    
    fs = 48000
    N = 960  # Khung 20ms chuẩn RNNoise (960 mẫu)
    t = np.arange(N) / fs
    
    # ==========================================
    # TRIỂN KHAI TOÁN HỌC VORBIS
    # ==========================================
    # w(n) = sin(pi/2 * sin^2(pi * (n + 0.5) / N))
    n = np.arange(N)
    vorbis_window = np.sin((np.pi / 2) * (np.sin(np.pi * (n + 0.5) / N))**2)
    
    # ==========================================
    # MÔ PHỎNG TÍN HIỆU RÒ RỈ
    # ==========================================
    # Cố tình tạo một sóng 400.1 Hz chạy không khít viền của 960 mẫu (Tương đương tiếng nói thực tế)
    # Lệch pha một chút so với lưới FFT để dễ quan sát Thảm họa Rò rỉ
    f0 = 400.1 
    clean_wave = np.sin(2 * np.pi * f0 * t) * 5000
    
    # Ép màng lọc dốc
    windowed_wave = clean_wave * vorbis_window
    
    # Biến đổi FFT 960 điểm
    fft_raw = np.abs(np.fft.rfft(clean_wave, n=N))
    fft_raw_db = 20 * np.log10(fft_raw + 1e-12)
    
    fft_win = np.abs(np.fft.rfft(windowed_wave, n=N))
    fft_win_db = 20 * np.log10(fft_win + 1e-12)
    
    freqs = np.fft.rfftfreq(N, d=1/fs)

    # ==========================================
    # VẼ BIỂU ĐỒ SLIDE
    # ==========================================
    plt.style.use('default')
    fig, axs = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle("Phẫu thuật Toán học Cửa Sổ Vorbis (Tiền xử lý FFT 960 Bins)", fontsize=18, fontweight='bold', color='#1565C0')
    plt.subplots_adjust(hspace=0.45)

    # Đồ thị 1: Tín hiệu Thời gian
    axs[0].plot(t*1000, clean_wave, color='#D32F2F', linewidth=1.5, alpha=0.4, label="Tín hiệu Thô (Cắt gãy phéng ở hai lề tạo thành vách đá)")
    axs[0].plot(t*1000, windowed_wave, color='#2E7D32', linewidth=2, label="Đã ốp Cửa sổ Vorbis (Xả dốc mượt về 0V)")
    axs[0].plot(t*1000, vorbis_window * 5000, color='black', linestyle='dashed', label="Hình dáng Đường cong Cửa Sổ W(n)")
    axs[0].set_title("1. Miền Thời gian (Time Domain): Vuốt nhẵn Vết chém của Khung Mẩu 20ms", fontweight='bold')
    axs[0].set_ylabel("Biên độ (Voltage)")
    axs[0].set_xlabel("Thời gian (ms)")
    axs[0].legend(loc="upper right")
    axs[0].grid(True, linestyle='--', alpha=0.5)

    # Đồ thị 2: Hiện tượng Rò rỉ Phổ (Spectral Leakage)
    axs[1].plot(freqs, fft_raw_db, color='#D32F2F', linewidth=1.5, alpha=0.7, label="Phổ Thô (Méo mó và Tràn năng lượng đi khắp nơi do Tách Đứt Lề)")
    axs[1].plot(freqs, fft_win_db, color='#2E7D32', linewidth=2.5, label="Phổ Vorbis (Bén ngót, Tụ năng lượng khít rịt vào 1 Vạch)")
    axs[1].set_title("2. Miền Tần số (FFT): Khống chế Ác mộng Rò Rỉ Phổ (Spectral Leakage)", fontweight='bold')
    axs[1].set_xlim(0, 1500)  # Zoom in để quan sát Rò rỉ
    axs[1].set_ylim(-20, 140)
    axs[1].set_ylabel("Năng lượng (dB)")
    axs[1].set_xlabel("Tần số (Hz)")
    axs[1].legend(loc="upper right")
    axs[1].grid(True, linestyle='--', alpha=0.5)
    
    axs[1].text(800, 40, "Phổ Thô Lè Nhè\n(Méo mó, lừa hệ thống rằng có rất nhiều Tần số rác)", color='#D32F2F', fontweight='bold', fontsize=11)
    axs[1].text(480, 120, "Đỉnh Dao Cạo Vorbis\n(Ghim chuẩn 1 Tần số gốc: 400Hz)", color='#2E7D32', fontweight='bold', fontsize=11)

    # Đồ thị 3: Tính chất Tái tạo Perfect Reconstruction Overlap-Add
    half_N = N // 2 # 480 mẫu chồng lấn
    t_samples = np.arange(N)
    w1_sq = vorbis_window**2
    axs[2].plot(t_samples, w1_sq, color='#1976D2', linewidth=2.5, label="$W_1^2$ (Hệ số Bình phương của Khung Hiện Tại)")
    
    # Tạo một khung số 2 chồng lấn 50%
    w2 = np.zeros(N + half_N)
    w2[half_N:] = vorbis_window
    w2_sq_full = w2**2
    axs[2].plot(np.arange(half_N, half_N + N), w2_sq_full[half_N:], color='#F57C00', linewidth=2.5, label="$W_2^2$ (Hệ số Bình phương của Khung Tiếp Theo chập lấn)")
    
    # Tính Tổng 2 đường cong tại khúc giao thoa (Mẫu 480 -> 960)
    sum_w2 = w1_sq[half_N:] + w2_sq_full[half_N:N]
    axs[2].plot(np.arange(half_N, N), sum_w2, color='black', linewidth=4, linestyle='-', label="Tổng Chập $W_1^2 + W_2^2 = 1.0$ (Hoàn hảo tuyệt đối Không Khuyết Tật)")
    
    axs[2].set_title("3. Đoạn IFFT Syntheis: Định lý Tái tạo Hoàn hảo của Cửa sổ Ogg Vorbis khi Gối Đầu", fontweight='bold', color='#4A148C')
    axs[2].set_ylabel("Hệ Số Năng Lượng ($W^2$)")
    axs[2].set_xlabel("Chỉ số Giao thoa Mẩu Âm thanh (Samples)")
    axs[2].axvspan(half_N, N, color='#E1BEE7', alpha=0.3, label="Vùng Canh tác Chập Lấn (Overlapping 50%)")
    axs[2].set_ylim(0, 1.2)
    axs[2].legend(loc="center left")
    axs[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    filename = "rnnoise_vorbis_window_report.png"
    plt.savefig(filename, dpi=300)
    print(f"Đã xuất file Ảnh Slide Vorbis: {filename}")

if __name__ == "__main__":
    generate_vorbis_plot()
