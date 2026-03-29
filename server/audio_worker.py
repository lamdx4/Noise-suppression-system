import socket
import time
import wave
import os
import numpy as np
import pyaudio
from PyQt5 import QtCore
from audio_processor import RNNoiseWrapper, FRAME_SIZE

SAMPLE_RATE = 48000
CHANNELS = 1
UDP_IP = "0.0.0.0"
UDP_PORT = 12345
DLL_PATH = "./rnnoise.dll"

class AudioServer(QtCore.QObject):
    # Consolidated UI update signal (emitted at ~33 FPS)
    ui_update = QtCore.pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.running = True
        self.rnnoise = RNNoiseWrapper(DLL_PATH)
        
        # Runtime params
        self.bypass = False
        self.digital_gain = 1.0
        
        # PyAudio setup (Always in background thread)
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                channels=CHANNELS,
                                rate=SAMPLE_RATE,
                                output=True,
                                frames_per_buffer=FRAME_SIZE)
        
        # Socket setup
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((UDP_IP, UDP_PORT))
        self.sock.settimeout(1.0)
        
        # History buffers for visualization
        self.history_len = SAMPLE_RATE // 5 # 200ms
        self.raw_history = np.zeros(self.history_len, dtype=np.float32)
        self.clean_history = np.zeros(self.history_len, dtype=np.float32)
        
        # Buffers for long STFT if needed (we stick to frame-by-frame for speed)
        self.n_fft = 512
        self.hanning = np.hanning(FRAME_SIZE)
        
        # Recording state
        self.is_recording = False
        self.wav_raw = None
        self.wav_clean = None

    def start_recording(self):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        os.makedirs("recordings", exist_ok=True)
        self.wav_raw = wave.open(f"recordings/raw_{timestamp}.wav", 'wb')
        self.wav_raw.setnchannels(CHANNELS); self.wav_raw.setsampwidth(2); self.wav_raw.setframerate(SAMPLE_RATE)
        self.wav_clean = wave.open(f"recordings/clean_{timestamp}.wav", 'wb')
        self.wav_clean.setnchannels(CHANNELS); self.wav_clean.setsampwidth(2); self.wav_clean.setframerate(SAMPLE_RATE)
        self.is_recording = True

    def stop_recording(self):
        self.is_recording = False
        if self.wav_raw: self.wav_raw.close()
        if self.wav_clean: self.wav_clean.close()

    def run(self):
        packet_count = 0
        print(f"Background AudioWorker started on {UDP_IP}:{UDP_PORT}")
        
        while self.running:
            try:
                data, addr = self.sock.recvfrom(2048)
                if len(data) == FRAME_SIZE * 2:
                    t_start = time.perf_counter()
                    raw_audio = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                    
                    # Dầu vào RNNoise: Float32
                    if not self.bypass:
                        clean_audio, vad = self.rnnoise.process(raw_audio)
                    else:
                        clean_audio = raw_audio.copy()
                        vad = 0.0
                    
                    clean_audio *= self.digital_gain
                    t_ms = (time.perf_counter() - t_start) * 1000
                    
                    # Playback
                    playback_data = clean_audio.clip(-32768, 32767).astype(np.int16)
                    self.stream.write(playback_data.tobytes())
                    
                    if self.is_recording:
                        self.wav_raw.writeframes(data)
                        self.wav_clean.writeframes(playback_data.tobytes())
                    
                    packet_count += 1
                    
                    # Cập nhật buffer trượt cho GUI (Normalized)
                    norm_raw = raw_audio / 32768.0
                    norm_clean = clean_audio / 32768.0
                    
                    self.raw_history = np.roll(self.raw_history, -len(norm_raw))
                    self.raw_history[-len(norm_raw):] = norm_raw
                    self.clean_history = np.roll(self.clean_history, -len(norm_clean))
                    self.clean_history[-len(norm_clean):] = norm_clean
                    
                    # --- FPS THROTTLING (33 FPS) ---
                    if packet_count % 3 == 0:
                        # 1D FFT cho Spectrum (Vô cùng nhẹ)
                        f_raw = np.abs(np.fft.rfft(norm_raw * self.hanning, n=self.n_fft))
                        db_raw = 20 * np.log10(f_raw / 256 + 1e-6)
                        
                        f_clean = np.abs(np.fft.rfft(norm_clean * self.hanning, n=self.n_fft))
                        db_clean = 20 * np.log10(f_clean / 256 + 1e-6)
                        
                        # Emit gói dữ liệu duy hướng
                        self.ui_update.emit({
                            'wave_raw': self.raw_history.copy(),
                            'wave_clean': self.clean_history.copy(),
                            'db_raw': db_raw,
                            'db_clean': db_clean,
                            'vad': vad,
                            'proc_time': t_ms
                        })

            except socket.timeout: continue
            except Exception as e:
                if self.running: print(f"Worker Error: {e}")
                break

    @QtCore.pyqtSlot()
    def stop(self):
        print("Stopping AudioWorker...")
        self.running = False
        self.stop_recording()
        
        # Giải phóng tài nguyên Audio
        try:
            if hasattr(self, 'stream'):
                self.stream.stop_stream()
                self.stream.close()
            if hasattr(self, 'p'):
                self.p.terminate()
        except: pass
            
        # Giải phóng Socket
        try:
            if hasattr(self, 'sock'):
                self.sock.close()
        except: pass
            
        # Giải phóng RNNoise (Kích hoạt __del__ trong audio_processor)
        self.rnnoise = None
        print("AudioWorker resources released.")

