import numpy as np
import matplotlib.pyplot as plt
import os

def generate_leakage_plot():
    print("Khởi tạo mô phỏng Thảm họa Khung cắt đứt gãy & Rò rỉ phổ...")

    # Tạo thư mục output nếu chưa có
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "img_mocks")
    os.makedirs(output_dir, exist_ok=True)
    
    fs = 48000  # Tần số lấy mẫu
    N = 960     # Kích thước khung 20ms
    t = np.arange(N) / fs
    
    # 1. Tín hiệu gốc: Sóng Sine 400.1 Hz (Cố tình làm lẻ để không chẵn chu kỳ, tạo đứt gãy gắt)
    f0 = 400.1 
    
    # Cắt Bạo Biện (Rectangular Window - Vách đá)
    # Lùi t lại một chút để không bắt đầu từ 0 tuyệt đối -> Cắt ngang thân sóng
    abrupt_chunk = np.sin(2 * np.pi * f0 * (t + 0.005)) 
    
    # Cắt Vorbis (Vuốt mượt mép)
    n = np.arange(N)
    vorbis_window = np.sin((np.pi / 2) * (np.sin(np.pi * (n + 0.5) / N))**2)
    smooth_chunk = abrupt_chunk * vorbis_window
    
    # 2. Biến đổi FFT để soi Cõi Tần Số
    # Dùng Zero-padding (N_fft lớn) để nhìn thấy rõ Bãi Rác Sidelobes của sóng cắt Gãy
    N_fft = fs  # Độ phân giải 1Hz
    fft_abrupt = np.abs(np.fft.rfft(abrupt_chunk, n=N_fft))
    fft_abrupt_db = 20 * np.log10(fft_abrupt / np.max(fft_abrupt) + 1e-12) # Chuẩn hóa về 0dB
    
    fft_smooth = np.abs(np.fft.rfft(smooth_chunk, n=N_fft))
    fft_smooth_db = 20 * np.log10(fft_smooth / np.max(fft_smooth) + 1e-12) # Chuẩn hóa về 0dB
    
    freqs = np.fft.rfftfreq(N_fft, d=1/fs)

    # 3. VẼ BIỂU ĐỒ BÁO CÁO CỰC NÉT
    plt.style.use('default')
    # Tùy chỉnh Font để hiển thị tiếng Việt mượt mà
    plt.rcParams['font.family'] = 'sans-serif'
    
    fig, axs = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle("Thảm Họa Đứt Gãy Thời Gian (Spectral Leakage) & Phép Màu Vorbis", fontsize=18, fontweight='bold', color='#B71C1C')
    plt.subplots_adjust(hspace=0.45)

    # Miền thời gian: Dải liên tục và Vết chém
    full_t = np.linspace(-5/1000, 25/1000, 1500)
    full_wave = np.sin(2 * np.pi * f0 * (full_t + 0.005))
    
    axs[0].plot(full_t*1000, full_wave, color='gray', linestyle='--', alpha=0.5, label="Dòng chảy Âm thanh Lịch sử & Tương lai (Khung Gốc Liền Mạch)")
    axs[0].plot(t*1000, abrupt_chunk, color='#D32F2F', linewidth=3, label="Đoạn 20ms Băng Chuyền Bị Chặt Đứt Gãy (Khung Cắt Vuông Góc)")
    axs[0].axvline(x=0, color='black', linestyle='-.', lw=2)
    axs[0].axvline(x=20, color='black', linestyle='-.', lw=2)
    
    # Highlight the sharp edges (Vertical Drop lines)
    axs[0].plot([0, 0], [0, abrupt_chunk[0]], color='red', lw=4, zorder=5)
    axs[0].plot([20, 20], [abrupt_chunk[-1], 0], color='red', lw=4, zorder=5)
    
    axs[0].text(0.5, 0.6, "Vách Đá Chết Chóc\n(Bị Chém Khúc Đứt Lìa)", color='black', fontweight='bold', fontsize=11, bbox=dict(facecolor='yellow', alpha=0.8))
    axs[0].text(15.5, abrupt_chunk[-1] - 0.4, "Vách Đá Chết Chóc\n(Tạo Đứt Gãy Lớn)", color='black', fontweight='bold', fontsize=11, bbox=dict(facecolor='yellow', alpha=0.8))
    
    axs[0].set_title("1. Thảm Họa Miền Thời Gian: Cắt Khung 20ms Thô Bạo Bằng Cửa Sổ Chữ Nhật", fontweight='bold', color='#D32F2F')
    axs[0].set_ylabel("Biên Độ Sóng")
    axs[0].set_xlabel("Thời Gian (ms)")
    axs[0].legend(loc="lower center")
    axs[0].grid(True, linestyle='--', alpha=0.5)

    # Miền thời gian: Liều thuốc Vorbis
    axs[1].plot(t*1000, abrupt_chunk, color='#FFCDD2', linewidth=2, label="Vết Thương Đứt Gãy Gốc")
    axs[1].plot(t*1000, vorbis_window, color='black', linestyle=':', lw=2, label="Đường Dao Phẫu Thuật Vorbis")
    axs[1].plot(t*1000, smooth_chunk, color='#2E7D32', linewidth=3, label="Sóng Âm Đã Cuộn Tròn Mép Về 0 Volt (Bảo Toàn Chu Kỳ)")
    
    axs[1].annotate('Mép Khâu \nVề Điểm 0 An Toàn', xy=(0, 0), xytext=(2, -0.8), arrowprops=dict(facecolor='green', shrink=0.08), fontweight='bold', color='green', fontsize=11)
    axs[1].annotate('Mép Khâu \nVề Điểm 0 An Toàn', xy=(20, 0), xytext=(15, 0.6), arrowprops=dict(facecolor='green', shrink=0.08), fontweight='bold', color='green', fontsize=11)
    
    axs[1].set_title("2. Giải Pháp Cửa Sổ Vorbis (Vorbis Tapering): Vuốt Góc Giảm Áp Vách Đá Về Điểm Không", fontweight='bold', color='#2E7D32')
    axs[1].set_ylabel("Biên Độ Sóng")
    axs[1].set_xlabel("Thời Gian (ms)")
    axs[1].legend(loc="lower right")
    axs[1].grid(True, linestyle='--', alpha=0.5)

    # Miền Tần Số: Hậu Qủa Rò Rỉ Phổ
    # Phễu Rác Abrupt (Red)
    axs[2].plot(freqs, fft_abrupt_db, color='#D32F2F', linewidth=1.5, alpha=0.8, label='Lỗi Cắt Vuông Góc (Chữ Nhật): Tạp Âm Vỡ Vụn Rò Rỉ Khắp Nơi')
    axs[2].fill_between(freqs, fft_abrupt_db, -100, color='#FFCDD2', alpha=0.5)
    
    # Mũi Kim Vorbis (Green)
    axs[2].plot(freqs, fft_smooth_db, color='#2E7D32', linewidth=2.5, label='Vorbis: Bóp Chết Rác Âm -> Khối Phổ Gọn Gàng Sạch Bóng')
    axs[2].fill_between(freqs, fft_smooth_db, -100, color='#C8E6C9', alpha=0.7)
    
    axs[2].set_xlim(0, 1500)
    axs[2].set_ylim(-80, 10)
    axs[2].set_title("3. Bức Tranh Miền Tần Số FFT (Bảng Lệnh Sinh Tử AI): Rò Rỉ Phổ Gây Nhiễu Ảo Giác Mạng Nơ-Ron", fontweight='bold', color='#4A148C')
    axs[2].set_ylabel("Cường Độ Năng Lượng Phổ (dB)")
    axs[2].set_xlabel("Tần Số (Hz)")
    
    # Text Annotation inside Frequency plot
    axs[2].text(550, -10, "Bãi Rác Rò Rỉ Phổ (Tràn Lan -30dB)\nMạng AI bị Ảo Giác Tiếng Ồn", color='#B71C1C', fontweight='bold', fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='#B71C1C'))
    axs[2].text(20, -50, "Đỉnh Tần Số Người Nói Sắc Lẹm\nTiếng Rác Hai Bên Bị Ép Xuống Vực Sâu (-80dB)", color='#1B5E20', fontweight='bold', fontsize=11)
    
    axs[2].legend(loc="upper right")
    axs[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    filename = os.path.join(output_dir, "rnnoise_spectral_leakage_discontinuity_report.png")
    plt.savefig(filename, dpi=300, facecolor='white', bbox_inches='tight')
    print(f"Hoàn Thành: Tệp Visual Slide Đứt Gãy Phổ đã được lưu tại {filename}")

if __name__ == "__main__":
    generate_leakage_plot()
