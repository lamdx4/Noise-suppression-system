import socket
import pyaudio
import struct
import wave

# Cấu hình mạng
HOST = "0.0.0.0"
PORT = 3000

# Cấu hình âm thanh
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 8000
CHUNK = 8000

# Khởi tạo socket TCP server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print("Đang chờ ESP32 kết nối...")
conn, addr = server_socket.accept()
print(f"Kết nối từ {addr}")

# Khởi tạo PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK)

# Khởi tạo file WAV
wav_file = wave.open("res_audio.wav", "wb")
wav_file.setnchannels(CHANNELS)
wav_file.setsampwidth(p.get_sample_size(FORMAT))
wav_file.setframerate(RATE)

try:
    while True:
        # Nhận header tổng (little-endian)
        header = conn.recv(4)
        if not header:
            print("Không nhận được header tổng, kết nối bị ngắt.")
            break
        total_data_size = int.from_bytes(header, 'little')  # Sửa thành little-endian
        print(f"Đợi nhận tổng cộng {total_data_size} bytes")

        # Nhận dữ liệu từng đoạn
        total_data = b""
        while len(total_data) < total_data_size:
            # Nhận header đoạn (little-endian)
            chunk_header = conn.recv(4)
            if not chunk_header:
                print("Không nhận được header đoạn, kết nối bị ngắt.")
                break
            chunk_size = int.from_bytes(chunk_header, 'little')  # Sửa thành little-endian
            print(f"Đợi nhận đoạn {chunk_size} bytes")

            # Nhận dữ liệu đoạn
            chunk_data = b""
            while len(chunk_data) < chunk_size:
                packet = conn.recv(chunk_size - len(chunk_data))
                if not packet:
                    print("Không nhận đủ dữ liệu đoạn, kết nối bị ngắt.")
                    break
                chunk_data += packet

            total_data += chunk_data
            print(f"Nhận được đoạn {len(chunk_data)} bytes, tổng cộng {len(total_data)}/{total_data_size} bytes")

        # Kiểm tra và xử lý dữ liệu
        if len(total_data) == total_data_size:
            numbers = list(struct.unpack(f"{len(total_data) // 2}h", total_data))
            print(f"Numbers : {numbers}")
            print(f"Độ dài numbers: {len(numbers)}")
            stream.write(total_data)
            wav_file.writeframes(total_data)
        else:
            print("Không nhận đủ dữ liệu tổng, bỏ qua khung này.")

except KeyboardInterrupt:
    print("\nDừng server.")
finally:
    wav_file.close()
    stream.stop_stream()
    stream.close()
    p.terminate()
    conn.close()
    server_socket.close()