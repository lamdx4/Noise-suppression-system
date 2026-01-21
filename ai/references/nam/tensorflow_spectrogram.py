import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Cấu hình
SAMPLING_RATE = 10.0  # Hz
HOP_SIZE = 4           # số mẫu giữa các frame (frame_step)
WINDOW_SIZE = 16       # số mẫu mỗi cửa sổ (frame_length)
DURATION = 2.0         # thời gian tín hiệu (giây)
FREQ = 2.0             # tần số sóng sin (Hz)

# Tạo sóng sin
t = np.linspace(0, DURATION, int(SAMPLING_RATE * DURATION), endpoint=False)
x = np.sin(2 * np.pi * FREQ * t).astype(np.float32)

# STFT với TensorFlow
stft = tf.signal.stft(
    x,
    frame_length=WINDOW_SIZE,
    frame_step=HOP_SIZE,
    fft_length=WINDOW_SIZE
)
spectrogram = tf.abs(stft)

# Hiển thị spectrogram
plt.figure(figsize=(10, 4))
plt.imshow(
    tf.transpose(spectrogram).numpy(),
    aspect='auto',
    origin='lower',
    extent=[
        0,
        DURATION,
        0,
        SAMPLING_RATE / 2
    ]
)
plt.title('Spectrogram of 2Hz Sine Wave')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(label='Magnitude')
plt.tight_layout()
plt.show()
