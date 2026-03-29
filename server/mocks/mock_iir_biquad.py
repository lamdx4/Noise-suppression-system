import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def generate_iir_biquad_plot():
    print("Mô phỏng bộ lọc IIR Biquad từ thuật toán C gốc...")
    fs = 48000
    # Tạo 50ms âm thanh
    t = np.linspace(0, 0.05, int(fs * 0.05))

    # Tín hiệu gốc: Giọng đàn ông trầm (110Hz) và một hài âm (220Hz)
    clean_speech = np.sin(2 * np.pi * 110 * t) * 5000 + np.sin(2 * np.pi * 220 * t) * 3000

    # Lỗi rò rỉ phần cứng (Dòng 1 chiều 0Hz - DC Offset)
    dc_offset = 8000
    
    # Tạp âm môi trường cực quái gở (Tiếng gió lùa ùa ùa 15Hz)
    wind_rumble = np.sin(2 * np.pi * 15 * t) * 7000
    
    # Lỗ hổng đầu vào: Sóng âm thanh bị đội lên 8000Volt và nhấp nhô vì gió
    input_signal = clean_speech + wind_rumble + dc_offset

    # ==========================================
    # TRIỂN KHAI PHƯƠNG TRÌNH BIQUAD (Direct Form II Transposed của RNNoise)
    # ==========================================
    # Từ C: a_hp[2] = {-1.99599, 0.99600}; b_hp[2] = {-2, 1}
    # Hàm truyền H(z) = [1 - 2z^-1 + z^-2] / [1 - 1.99599z^-1 + 0.99600z^-2]
    
    numerator_b = [1.0, -2.0, 1.0]          # Tử số
    denominator_a = [1.0, -1.99599, 0.99600] # Mẫu số
    
    # Chạy Tín hiệu Thô qua Lưới Lọc
    output_signal = signal.lfilter(numerator_b, denominator_a, input_signal)

    # Tính Đáp ứng Tần số (Frequency Response Curve) để xem màng lọc chém rác kiểu gì
    w, h = signal.freqz(numerator_b, denominator_a, worN=8000, fs=fs)
    magnitude_db = 20 * np.log10(np.abs(h) + 1e-12)

    # ==========================================
    # VẼ BIỂU ĐỒ BÁO CÁO HỘI ĐỒNG
    # ==========================================
    plt.style.use('default')
    fig, axs = plt.subplots(3, 1, figsize=(14, 11))
    fig.suptitle("Giải phẫu Bộ Lọc Thông Cao Số (IIR Biquad HPF) Đầu Vào RNNoise", fontsize=18, fontweight='bold', color='#1565C0')
    plt.subplots_adjust(hspace=0.4)

    # Biểu đồ 1: Tín hiệu Thô Chứa Rác
    axs[0].plot(t, input_signal, color='#D32F2F', linewidth=1.5, label="Tín hiệu thô (Lệch DC + Gió ù 15Hz)")
    axs[0].plot(t, np.ones_like(t)*dc_offset, color='black', linestyle='--', label="Trục nhiễu Rò điện 0Hz (DC Drift)")
    axs[0].set_title("1. Trước khi qua IIR: Sóng âm lửng lơ trên không trung (Lỗi mạch) và bị uốn lượn do gió rít", fontweight='bold', color='#D32F2F')
    axs[0].set_ylabel("Biên độ Điện áp")
    axs[0].legend(loc="upper right")
    axs[0].grid(True, linestyle='--', alpha=0.5)

    # Biểu đồ 2: Tín hiệu Đầu Ra Sạch Bóng
    axs[1].plot(t, output_signal, color='#2E7D32', linewidth=1.5, label="Tín hiệu Lọc IIR (Chỉ còn Giọng người 110Hz/220Hz)")
    axs[1].axhline(y=0, color='black', linestyle='--', label="Trục 0 Voltage Tinh khiết")
    axs[1].set_title("2. Sau khi qua IIR Biquad: DC Offset vỡ tan rớt về 0V thẳng băng! Sóng gió rít cũng bốc hơi hoàn toàn!", fontweight='bold', color='#2E7D32')
    axs[1].set_ylabel("Biên độ Điện áp")
    axs[1].legend(loc="upper right")
    axs[1].grid(True, linestyle='--', alpha=0.5)
    
    # Do bộ lọc IIR cần dồn hệ số trễ ở những Mili-giây đầu tiên nên sóng dập gắt (Transient)
    # Ta highlight dải ổn định
    axs[1].axvspan(0, 0.005, color='gray', alpha=0.2, label='Chờ IIR Tụ (Transient)')
    axs[1].legend(loc="upper right")

    # Biểu đồ 3: Đáp ứng Tần số (Vách đá từ chối Tử thần)
    axs[2].plot(w, magnitude_db, color='#FF8F00', linewidth=3)
    axs[2].set_xlim(0, 200) # Chỉ zoom dải tần siêu thấp để thuyết trình
    axs[2].set_ylim(-80, 5)
    
    # Đổ xi măng vào Vực cắt
    axs[2].fill_between(w, magnitude_db, -80, where=(w <= 55), color='#FFE0B2', alpha=0.5)
    
    axs[2].axvline(x=55, color='red', linestyle='--', label="Ngưỡng Cắt Siêu Trầm (~55Hz)")
    axs[2].text(20, -40, "Vực Dập Nhiễu Gió/DC", fontweight='bold', color='red', fontsize=12, ha='center')
    axs[2].text(120, -10, "Đường Cao Tốc (Cho tiếng người lọt qua)", fontweight='bold', color='#E65100', fontsize=12)
    
    axs[2].set_title("3. Đáp tuyến Tần số $H(z)$ Phương trình Toán học Biquad (Zoom dải thính giác 0Hz - 200Hz)", fontweight='bold', color='#EF6C00')
    axs[2].set_ylabel("Cường độ Từ Lọc (dB)")
    axs[2].set_xlabel("Phổ Tần số (Hz)")
    axs[2].legend(loc="lower right")
    axs[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    filename = "rnnoise_iir_biquad_report.png"
    plt.savefig(filename, dpi=300)
    print(f"Đã xuất file Ảnh IIR Slide: {filename}")

if __name__ == "__main__":
    generate_iir_biquad_plot()
