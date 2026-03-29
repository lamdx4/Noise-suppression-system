import numpy as np
import matplotlib.pyplot as plt
import os

def generate_overlap_plot():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "img_mocks")
    os.makedirs(output_dir, exist_ok=True)
    
    fs = 48000
    N = 960  # 20ms
    HOP = 480 # 10ms
    t = np.arange(N * 2) / fs # Total time 40ms to show overlap
    
    f0 = 200 # A low frequency for visual clarity

    # The original continuous audio
    original_audio = np.sin(2 * np.pi * f0 * t)
    
    # Vorbis window function
    n = np.arange(N)
    vorbis = np.sin((np.pi / 2) * (np.sin(np.pi * (n + 0.5) / N))**2)

    # Frame 1: from 0 to 20ms
    frame1 = original_audio[0:N] * vorbis
    
    # Frame 2: from 10ms to 30ms (Starts 10ms later = 480 samples)
    frame2 = original_audio[HOP:HOP+N] * vorbis
    
    # Pure Window shapes for visualization (No Audio)
    win1_full = np.zeros_like(t)
    win1_full[0:N] = vorbis
    
    win2_full = np.zeros_like(t)
    win2_full[HOP:HOP+N] = vorbis
    
    # Superposition (The Sum)
    sum_windows = win1_full + win2_full
    
    plt.style.use('default')
    plt.rcParams['font.family'] = 'sans-serif'
    
    fig, axs = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle("Giải Phẫu: Tại Sao Vorbis Lại Cần Gối Đầu Nhau 50% (Overlap)?", fontsize=18, fontweight='bold', color='#1565C0')
    plt.subplots_adjust(hspace=0.45)

    # Plot 1: The problem with a single Vorbis window
    axs[0].plot(t[:N]*1000, original_audio[:N], color='gray', linestyle='--', label="Sóng âm thực tế")
    axs[0].plot(t[:N]*1000, frame1, color='#D32F2F', linewidth=3, label="Lỗi: Âm lượng bị bóp nghẹt 2 đầu về 0 Volt")
    axs[0].axvline(x=10, color='black', linestyle=':', label='Đỉnh Volume (10ms)')
    axs[0].text(2, -0.8, "Điếc Vị Trí 1\n(0ms)", color='#D32F2F', fontweight='bold', bbox=dict(facecolor='yellow', alpha=0.8))
    axs[0].text(15, -0.8, "Điếc Vị Trí 2\n(20ms)", color='#D32F2F', fontweight='bold', bbox=dict(facecolor='yellow', alpha=0.8))
    axs[0].set_title("1. Thảm Họa Nếu Chỉ Dùng 1 Khung Vorbis Đơn Độc (Volume Lúc To Lúc Nhỏ)", fontweight='bold', color='#D32F2F')
    axs[0].set_ylabel("Biên Độ")
    axs[0].set_xlabel("Thời Gian (ms)")
    axs[0].legend(loc="upper right")
    axs[0].grid(True, linestyle='--', alpha=0.5)

    # Plot 2: The Two Overlapping Windows
    axs[1].plot(t*1000, win1_full, color='#1976D2', linewidth=3, label="Màng Vorbis KHUNG 1 (0 -> 20ms)")
    axs[1].plot(t*1000, win2_full, color='#388E3C', linewidth=3, label="Màng Vorbis KHUNG 2 (10 -> 30ms)")
    axs[1].fill_between(t*1000, win1_full, color='#1976D2', alpha=0.2)
    axs[1].fill_between(t*1000, win2_full, color='#388E3C', alpha=0.2)
    axs[1].axvline(x=10, color='red', linestyle='--', lw=2, label="Trục Giao Thoa Vàng (10ms)")
    
    axs[1].text(10.5, 0.4, "Chóp Đỉnh Khung 1 (Tỉ lệ 100% Volume)\n+ Mép Trũng Khung 2 (Tỉ lệ 0% Volume)", color='black', fontweight='bold', bbox=dict(facecolor='white', alpha=0.9))
    axs[1].text(15.5, 0.5, "Ngã Tư Chéo 50/50\n(Mỗi Khung Đóng Góp Nửa Mạng)", color='black', fontweight='bold', bbox=dict(facecolor='white', alpha=0.9))
    
    axs[1].set_title("2. Giải Cứu Gối Đầu: Khung 2 Bước Lên Võ Đài Sớm Hơn Giữa Hiệp", fontweight='bold', color='#1976D2')
    axs[1].set_ylabel("Hệ số Volume (0.0 -> 1.0)")
    axs[1].set_xlabel("Thời Gian (ms)")
    axs[1].set_xlim(0, 30)
    axs[1].legend(loc="upper right")
    axs[1].grid(True, linestyle='--', alpha=0.5)

    # Plot 3: Perfect Reconstruction
    axs[2].plot(t*1000, sum_windows, color='#F57C00', linewidth=4, label="TỔNG VOLUME (Khung 1 + Khung 2)")
    
    axs[2].axhline(y=1.0, color='gray', linestyle='--', label="Ngưỡng Liền Mạch Hoàn Hảo (1.0)")
    axs[2].fill_between(t*1000, sum_windows, 0, where=(t*1000 >= 10) & (t*1000 <= 20), color='#FFE0B2', alpha=0.8)
    
    axs[2].text(13, 0.5, "VÙNG PHỤC HỒI\nHOÀN HẢO\n(Perfect Reconstruction)", color='#E65100', fontweight='bold', fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.9, edgecolor='#E65100'))
    
    axs[2].set_title("3. Hợp Thể Giao Thoa: Tổng 2 Khung Vorbis Luôn Bằng Đúng 1 Đường Cố Định (1.0)", fontweight='bold', color='#E65100')
    axs[2].set_ylabel("Tổng Năng Lượng")
    axs[2].set_xlabel("Thời Gian (ms)")
    axs[2].set_xlim(0, 30)
    axs[2].legend(loc="lower right")
    axs[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    filename = os.path.join(output_dir, "vorbis_overlap_add_explanation.png")
    plt.savefig(filename, dpi=300, facecolor='white', bbox_inches='tight')
    print(f"Hoàn Thành Visual Overlap: {filename}")

if __name__ == "__main__":
    generate_overlap_plot()
