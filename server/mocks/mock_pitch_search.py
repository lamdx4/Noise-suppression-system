import numpy as np
import matplotlib.pyplot as plt
import os

# ================= 1. CẤU HÌNH THÔNG SỐ VẬT LÝ =================
fs = 48000
f0 = 150 # Tần số cơ bản của giọng Thanh Quản (Nam - 150Hz)
true_lag = int(fs / f0) # Độ trễ hoàn hảo lý thuyết: 48000/150 = 320 Mẫu
hist_size = 1800 # Băng chuyền lịch sử mô phỏng (Dài đủ cho 960 + 768)
frame_size = 960 # Kích thước lăng kính Hiện Tại (20ms)

# ================= 2. TẠO HÌNH SÓNG CỔ HỌNG (VOICE & NOISE) =================
t = np.arange(hist_size) / fs
voice = np.zeros_like(t)
# Tổng hợp 6 sóng Hài Âm để tạo ra cấu trúc đỉnh núi lặp lại phức tạp
for k in range(1, 7): 
    voice += (1.0 / np.sqrt(k)) * np.sin(2 * np.pi * k * f0 * t + np.pi/3 * k)

np.random.seed(99)
noise = np.random.normal(0, 0.5, size=t.shape) # Rác quạt máy xen vào tiếng nói
signal = voice + noise

# ================= 3. THUẬT TOÁN DÒ PITCH (CROSS-CORRELATION rnn_pitch_search) =================
# Cắt thớt Hiện Tại 960 mẫu (Phía cuối cùng bên phải của Cột Thời gian)
x_current = signal[-frame_size:]

# Rải lăng kính trượt từ độ trễ 60 đến 768 Mẫu (Theo Mốc kẹp C của RNNoise)
lags = np.arange(60, 768)
xcorr = np.zeros(len(lags))

# Quét khổ sai Tích Vô Hướng (Chà xát 2 mặt thớt với nhau)
for i, lag in enumerate(lags):
    start_idx = hist_size - frame_size - lag
    end_idx = hist_size - lag
    y_past = signal[start_idx:end_idx]
    xcorr[i] = np.dot(x_current, y_past) # Tính Tổng Tương Quan (Dot Product)

# Truy Tìm Tọa độ Nhỉnh Nhất Đọt Đỉnh
best_index = np.argmax(xcorr)
best_lag = lags[best_index]

# Bốc Hốt Nguyên Đai Kiện Mảnh Quá Khứ Thắng Cuộc
best_start = hist_size - frame_size - best_lag
best_end = hist_size - best_lag
y_best = signal[best_start:best_end]

# ================= VẼ BÁO CÁO CỰC NÉT =================
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'

fig, axs = plt.subplots(3, 1, figsize=(14, 12))
fig.suptitle("Mô Phỏng Trực Quan Thuật Toán Dò Tìm Chu Kỳ Pitch Trên Miền Thời Gian (rnn_pitch_search)", fontsize=18, fontweight='bold', color='#283593')
plt.subplots_adjust(hspace=0.45)

# BẢNG 1: HIỆN TRƯỜNG DÒ KÝ ỨC (BĂNG CHUYỀN LỊCH SỬ)
axs[0].plot(np.arange(hist_size), signal, color='gray', alpha=0.5, label="Tín hiệu bộ đệm lịch sử (pitch_buf)")
# Tô sáng Cục Hiện tại (Đầu bên phải)
axs[0].plot(np.arange(hist_size - frame_size, hist_size), x_current, color='#D32F2F', lw=2, label="Khung thời gian phân tích hiện tại (X[n], Độ dài 960 mẫu)")
# Tô sáng Cục Tương Lai Thắng Cuộc
axs[0].plot(np.arange(best_start, best_end), y_best, color='#1976D2', lw=2, label=f"Khung tín hiệu trễ tương quan tối ưu (\u03C4 = {best_lag})")

# Vẽ Mũi tên Rẽ lùi Thời gian
axs[0].annotate(f'Phép dịch thời gian trượt\n(Sliding Window \u03C4 = {best_lag} mẫu)', xy=(best_end, 3), xytext=(hist_size - frame_size, 3.5),
                arrowprops=dict(facecolor='black', shrink=0.01, width=2, headwidth=8), ha='center', fontweight='bold')

axs[0].set_title("1. Cấu Trúc Bộ Đệm Lịch Sử (Pitch Buffer): Lưu Trữ 1728 Mẫu Tín Hiệu PCM Chưa Qua Cửa Sổ Hanning/Vorbis", fontweight='bold', color='#1A237E')
axs[0].set_ylim(-4, 5)
axs[0].set_xlim(0, hist_size)
axs[0].set_ylabel("Biên độ (Amplitude)")
axs[0].legend(loc="upper left")
axs[0].grid(True, linestyle='--', alpha=0.5)

# BẢNG 2: MÁY QUÉT ĐOT ĐỈNH TƯƠNG QUAN
axs[1].plot(lags, xcorr, color='#F57C00', linewidth=2.5, label="Đồ thị Tương Quan Chéo (Cross-Correlation Score)")
axs[1].fill_between(lags, xcorr, 0, color='#FFE0B2', alpha=0.5)

# Đóng Đinh Đỉnh Chóp Cao Nhất
axs[1].plot(best_lag, xcorr[best_index], marker='o', markersize=10, color='red')
axs[1].annotate(f'Cực Đại Toàn Cục (Global Maximum)\npitch_index = {best_lag}', xy=(best_lag, xcorr[best_index]), xytext=(best_lag+30, xcorr[best_index]-1000),
                arrowprops=dict(facecolor='red', shrink=0.05), fontweight='bold', color='red')

# Đóng Đinh Cái Bẫy Nhảy Quãng (Octave Doubling)
axs[1].plot(best_lag * 2, xcorr[abs(lags - best_lag * 2).argmin()], marker='x', markersize=10, color='purple')
axs[1].annotate("Cực đại địa phương / Lỗi Nhảy Quãng 8 (Octave Error)\n(Bị loại trừ bởi thuật toán rnn_pitch_search)", xy=(best_lag*2, xcorr[abs(lags - best_lag * 2).argmin()]), xytext=(best_lag*2 - 80, xcorr[abs(lags - best_lag * 2).argmin()]+2000),
                arrowprops=dict(facecolor='purple', shrink=0.05), fontweight='bold', color='purple')

axs[1].set_title("2. Đồ Thị Tương Quan Phân Tích (Cross-Correlation Function): Đánh Giá Mức Độ Đồng Pha Theo Độ Trễ \u03C4", fontweight='bold', color='#E65100')
axs[1].set_xlim(60, 768)
axs[1].set_ylabel("Mức độ Tương Quan")
axs[1].set_xlabel("Độ trễ \u03C4 (Theo thang mẫu PCM)")
axs[1].legend(loc="upper right")
axs[1].grid(True, linestyle='--', alpha=0.5)

# BẢNG 3: BỮA TIỆC CHẠM PHA (KHỚP RẬP KHUÔN)
# Chỉ zoom in vào khúc 200 mẫu đầu của quá trình khớp để nhìn Mắt Pha Răng Cưa rõ nét
zoom = 300 
axs[2].plot(np.arange(zoom), x_current[:zoom], color='#D32F2F', linewidth=3, alpha=0.6, label="Tín hiệu Hiện tại X[n]")
axs[2].plot(np.arange(zoom), y_best[:zoom], color='#1976D2', linewidth=1.5, linestyle='-', label=f"Tín hiệu Quá khứ tối ưu P[n] (\u03C4 = {best_lag})")

# Vẽ đường nét đứt Chạm Khớp Góc
for peak in [45, 145, 245]: # Approximate peaks
    axs[2].axvline(x=peak, color='green', linestyle=':', lw=2, alpha=0.8)

axs[2].annotate('Đồng Bộ Chu Kỳ Hài Âm Hoàn Hảo (Harmonic Phase Match)\nTạp âm nền không tuần hoàn nằm ngoài pha hệ thống', xy=(145, 2.5), xytext=(170, 3.5), arrowprops=dict(facecolor='green', shrink=0.05), fontweight='bold', color='green', fontsize=12)

axs[2].set_title("3. Phân Tích Sự Đồng Bộ Pha (Phase Alignment): Đối Chiếu Tín Hiệu X[n] Hiện Tại Và Tín Hiệu Lịch Sử P[n]", fontweight='bold', color='#004D40')
axs[2].set_ylim(-4, 5)
axs[2].set_xlim(0, zoom)
axs[2].set_ylabel("Biên độ (Amplitude)")
axs[2].set_xlabel("Chỉ số mẫu (Sample Index) [0 : 300]")
axs[2].legend(loc="upper right")
axs[2].grid(True, linestyle='--', alpha=0.5)

# Lưu Mạch Cảm Xúc
output_dir = os.path.join(os.path.dirname(__file__), "img_mocks")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "rnnoise_pitch_search_tracker.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Cập nhật thành công: {output_path}")
