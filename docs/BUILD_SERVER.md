# Hướng dẫn Biên dịch và Chạy RNNoise Audio Server

Tài liệu này hướng dẫn cách build thư viện xử lý và khởi chạy dashboard giám sát trên Windows.

## 1. Yêu cầu hệ thống

- **Trình biên dịch**: Clang (LLVM)
- **Môi trường**: Python 3.x (cần các thư viện: `numpy`, `PyQt5`, `pyqtgraph`, `pyaudio`)

## 2. Biên dịch Shared Library (DLL)

Chạy lệnh sau để tạo file `rnnoise.dll` với tối ưu hóa AVX2 (dành cho CPU hiện đại):

```powershell
clang -shared -O3 -mavx2 -o server/rnnoise.dll `
    firmware/references/rnnoise-0.2/rnnoise-0.2/src/denoise.c `
    firmware/references/rnnoise-0.2/rnnoise-0.2/src/nnet.c `
    firmware/references/rnnoise-0.2/rnnoise-0.2/src/rnn.c `
    firmware/references/rnnoise-0.2/rnnoise-0.2/src/pitch.c `
    firmware/references/rnnoise-0.2/rnnoise-0.2/src/kiss_fft.c `
    firmware/references/rnnoise-0.2/rnnoise-0.2/src/celt_lpc.c `
    firmware/references/rnnoise-0.2/rnnoise-0.2/src/rnnoise_data.c `
    firmware/references/rnnoise-0.2/rnnoise-0.2/src/rnnoise_tables.c `
    firmware/references/rnnoise-0.2/rnnoise-0.2/src/nnet_default.c `
    firmware/references/rnnoise-0.2/rnnoise-0.2/src/parse_lpcnet_weights.c `
    -I firmware/references/rnnoise-0.2/rnnoise-0.2/include `
    -D WIN32 -D RNNOISE_BUILD -D DLL_EXPORT
```

## 3. Khởi chạy Server Dashboard

Sau khi biên dịch thành công file DLL, khởi chạy Dashboard bằng lệnh:

```powershell
python server/realtime_server.py
```

## 4. Cấu hình Kết nối

Cập nhật IP của Workstation vào code ESP32 (`main.cpp`):

- **Server IP**: 192.168.1.12
- **Port**: 12345
