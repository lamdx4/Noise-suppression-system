import numpy as np
import matplotlib.pyplot as plt
import re


def read_fft_data(file_path):
    frames = []
    current_frame = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            if line.startswith('Frame'):
                if current_frame:
                    frames.append(np.array(current_frame))
                    current_frame = []
                continue

            if not line:
                continue

            parts = re.split(r'\s+', line)
            if len(parts) >= 2:
                try:
                    real = float(parts[0])
                    imag = float(parts[1])
                    current_frame.append(complex(real, imag))
                except ValueError:
                    continue

    if current_frame:
        frames.append(np.array(current_frame))

    return np.array(frames)


# Thông số hệ thống
SAMPLING_RATE = 10.0  # Hz
HOP_SIZE = 4  # samples
WINDOW_SIZE = 16  # samples

# Đọc dữ liệu
fft_frames = read_fft_data('spectrogarm_arduino.txt')

# Tính toán các thông số
num_frames = fft_frames.shape[0]
frame_duration = HOP_SIZE / SAMPLING_RATE
total_duration = num_frames * frame_duration

# Tính toán bins tần số (chỉ lấy nửa đầu do tính đối xứng)
freq_bins = np.fft.rfftfreq(WINDOW_SIZE, 1 / SAMPLING_RATE)
num_freq_bins = len(freq_bins)

# Tính toán biên độ và chuẩn hóa kích thước
magnitude = np.abs(fft_frames)[:, :num_freq_bins]  # Chỉ lấy các bins tần số cần thiết

# Tạo figure
plt.figure(figsize=(12, 6))

# Cách 1: Sử dụng imshow (đơn giản, tự động căn chỉnh)
plt.imshow(magnitude.T,  # Chuyển vị để time là trục x
           aspect='auto',
           origin='lower',
           extent=[0, total_duration, 0, freq_bins[-1]],
           cmap='viridis')

# Cách 2: Sử dụng pcolormesh (chính xác hơn)
# time_edges = np.linspace(0, total_duration, num_frames + 1)
# freq_edges = np.linspace(0, freq_bins[-1], num_freq_bins + 1)
# plt.pcolormesh(time_edges, freq_edges, magnitude.T,
#               shading='auto', cmap='viridis')

plt.colorbar(label='Magnitude')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram Arduino')

# Giới hạn dải tần số để tập trung vào vùng quan tâm
plt.ylim(0, 5)  # 0-100Hz để thấy rõ tín hiệu 2Hz

plt.tight_layout()
plt.show()