import numpy as np
import matplotlib.pyplot as plt
import os

# --- MÔ PHỎNG LỌC LƯỢC (COMB FILTER) TRONG RNNOISE ---
# Thuật toán: X_sạch = X + alpha * P
# Dựa trên lý thuyết tín hiệu Tần số Nyquist

fs = 48000
f0 = 200 # Tần số gốc Thanh Quản (200Hz) - Giọng Nam trầm
T0 = int(fs / f0) # Chu kỳ thời gian trễ Tau (240 mẫu)
alpha = 0.85 # Hệ số Van Điều Tiết Pitch Gain Từ AI (Rất Mạnh)

# Trục Tần Số dải từ 0 đến 1500Hz
f = np.linspace(0, 1500, 3000)

# 1. TẠO HÌNH PHỔ X HIỆN TẠI (Noisy Spectrum) - Tăng nền rác để dễ thấy hiệu ứng
# Bơm các cột Hài Âm Giọng Người
S_power = np.zeros_like(f)
for harm in [200, 400, 600, 800, 1000, 1200, 1400]:
    peak_power = 10**(30.0 / 10) # Signal 30dB (Vừa đủ nhô khỏi rác)
    S_power += peak_power * np.exp(-((f - harm)/6)**2)

# Nền Rác Tạp Âm Dày Đặc
np.random.seed(42)
N_power_db = np.random.normal(24, 2, size=f.shape) # Rác 24dB
N_power = 10**(N_power_db/10)

# Tổng Hợp Hiện Tại X
X_power = S_power + N_power
X_noisy_dB = 10 * np.log10(X_power + 1e-12)

# 2. BỘ LỌC RĂNG LƯỢC (COMB FILTER) MÔ HÌNH VẬT LÝ
tau_sec = T0 / fs
H_complex = 1 + alpha * np.exp(-1j * 2 * np.pi * f * tau_sec)
S_clean_power = S_power * (np.abs(H_complex)**2)

# Rác ngẫu nhiên độc lập 2 hệ X và P
N_clean_power = N_power * (1 + alpha**2)

# Chuẩn Hóa Phương Sai Theo B4
norm_factor = 1 + alpha**2
X_clean_power = (S_clean_power + N_clean_power) / norm_factor
X_clean_dB = 10 * np.log10(X_clean_power + 1e-12)

# Đồ thị Hàm Truyền Toán Học
H_dB_Response = 10 * np.log10(np.abs(H_complex)**2 / norm_factor + 1e-12)

# ================= VẼ BÁO CÁO CỰC NÉT =================
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'

fig, axs = plt.subplots(3, 1, figsize=(14, 12))
fig.suptitle("Tuyệt Kỹ Khử Nhiễu Răng Lược (Pitch Comb Filter): Ép Rác Cũ Đứng Yên - Phóng Đỉnh Giọng Lên", fontsize=18, fontweight='bold', color='#1565C0')
plt.subplots_adjust(hspace=0.45)

# BẢNG 1: HIỆN TRẠNG TỒI TỆ
axs[0].plot(f, X_noisy_dB, color='#D32F2F', linewidth=1.5, alpha=0.9, label="Cấu Trúc Phổ Hiện Tại Bị Nhiễu Nặng (X_noisy)")
axs[0].fill_between(f, X_noisy_dB, 10, color='#FFCDD2', alpha=0.4)
for harm in [200, 400, 600, 800, 1000, 1200, 1400]:
    axs[0].axvline(x=harm, color='black', linestyle=':', lw=1, alpha=0.4)

axs[0].set_title("1. Khởi Điểm: Các Đỉnh Giọng Yếu Ớt (30dB) Đang Bị Cạnh Tranh Rất Khốc Liệt Trực Tiếp Từ Lớp Mây Rác (24dB)", fontweight='bold', color='#B71C1C')
axs[0].set_ylim(15, 36)
axs[0].set_xlim(0, 1500)
axs[0].set_ylabel("Cường Độ (dB)")
axs[0].legend(loc="upper right")
axs[0].grid(True, linestyle='--', alpha=0.5)

# BẢNG 2: ĐÁP ỨNG TRUYỀN HỆ THỐNG
axs[1].plot(f, H_dB_Response, color='#FF8F00', linewidth=3, label="Hàm Truyền Lọc Lược Alpha = 0.85")
axs[1].fill_between(f, H_dB_Response, -15, color='#FFECB3', alpha=0.5)
axs[1].annotate('Mũi Khoan Trúng Đích (+3dB)', xy=(400, H_dB_Response[np.argmax(H_dB_Response[:1000])]-0.5), xytext=(250, -2), arrowprops=dict(facecolor='black', shrink=0.05), fontweight='bold', color='#E65100')
axs[1].annotate('Vùng Chặn Rác Ảo (- Giữ Phương Sai)', xy=(900, H_dB_Response[np.argmin(H_dB_Response[:2400])]+2), xytext=(700, -8), arrowprops=dict(facecolor='black', shrink=0.05), fontweight='bold', color='#37474F')

axs[1].set_title("2. Cấu Hình Toán Học Lọc Lược Áp Mạch: Chỉ Dung Nạp Và Bơm Tín Hiệu Cho Răng Cưa Đồng Pha", fontweight='bold', color='#E65100')
axs[1].set_ylim(-15, 5)
axs[1].set_xlim(0, 1500)
axs[1].set_ylabel("Hệ Số Khuếch (dB)")
axs[1].legend(loc="upper right")
axs[1].grid(True, linestyle='--', alpha=0.5)

# BẢNG 3: SỰ SỐNG TÁI SINH
axs[2].plot(f, X_noisy_dB, color='#D32F2F', linewidth=2, linestyle='-', alpha=0.4, label="Vết Tích Rác Lũ Cũ Đè Bẹp Giọng (Màu Đỏ Mờ)")
axs[2].plot(f, X_clean_dB, color='#2E7D32', linewidth=2, label="Phổ Sạch Thực Tế (X_clean): Đỉnh Giọng Xanh Lục Vững Chãi Bứt Phá Khỏi Cơn Lũ Đỏ")
axs[2].fill_between(f, X_clean_dB, 10, color='#C8E6C9', alpha=0.6)

for harm in [200, 400, 600, 800, 1000, 1200, 1400]:
    axs[2].axvline(x=harm, color='black', linestyle=':', lw=1, alpha=0.4)

snr_boost = 10 * np.log10(((1+alpha)**2) / (1**2 + alpha**2))
axs[2].annotate(f'ĐỈNH HÀI ÂM BỨT PHÁ KHỎI RÁC!\nKhoảng cách Đỏ Cũ vs Xanh Mới \n(Tăng SNR {snr_boost:.1f}dB Định Vị Mũi Nhọn)', xy=(600, X_clean_dB[np.argmax((f>590)&(f<610))] - 1), xytext=(650, 31), arrowprops=dict(facecolor='green', shrink=0.05), fontweight='bold', color='green', fontsize=11)

axs[2].set_title("3. Bức Tranh Lột Xác (SỰ THẬT VẬT LÝ): Lớp Sương Mù Mãi Nhét Nguyên Mức Cũ... Nhưng Toàn Bộ Khối Giọng Nói Kịch Kim Rút CAO HƠN CŨ MỘT KHÚC!", fontweight='bold', color='#1B5E20')
axs[2].set_ylim(15, 36)
axs[2].set_xlim(0, 1500)
axs[2].set_ylabel("Cường Độ (dB)")
axs[2].set_xlabel("Tần Số (Hz)")
axs[2].legend(loc="upper right")
axs[2].grid(True, linestyle='--', alpha=0.5)

# Lưu Mạch Cảm Xúc
output_dir = os.path.join(os.path.dirname(__file__), "img_mocks")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "rnnoise_comb_filter_magic.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"XUYÊN TIM! Biểu đồ Lọc Răng Lược Pitch Filter đã đúc xong tại: {output_path}")
