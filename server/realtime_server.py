import sys
import socket
import threading
import numpy as np
import pyaudio
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets, QtGui
import ctypes
import time
import wave
import os

# --- Configuration ---
UDP_IP = "0.0.0.0"
UDP_PORT = 12345
SAMPLE_RATE = 48000
CHANNELS = 1
FRAME_SIZE = 480  # 10ms for 48kHz
DLL_PATH = "./rnnoise.dll"

# --- RNNoise C-Binding Wrapper ---
class RNNoiseWrapper:
    def __init__(self, dll_path):
        self.lib = ctypes.CDLL(dll_path)
        
        # Define function signatures
        self.lib.rnnoise_create.restype = ctypes.c_void_p
        self.lib.rnnoise_create.argtypes = [ctypes.c_void_p]
        
        self.lib.rnnoise_process_frame.restype = ctypes.c_float
        self.lib.rnnoise_process_frame.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        
        self.lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
        
        # Initialize state
        self.st = self.lib.rnnoise_create(None)
        
    def process(self, audio_data: np.ndarray):
        """Processes a frame of 480 samples. Input/Output: float32"""
        if len(audio_data) != FRAME_SIZE:
            return audio_data, 0.0
            
        in_ptr = audio_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        out_ptr = (ctypes.c_float * FRAME_SIZE)()
        
        vad_prob = self.lib.rnnoise_process_frame(self.st, out_ptr, in_ptr)
        
        processed_data = np.frombuffer(out_ptr, dtype=np.float32)
        return processed_data, vad_prob

    def __del__(self):
        if hasattr(self, 'st'):
            self.lib.rnnoise_destroy(self.st)

# --- Server Core ---
class AudioServer(QtCore.QObject):
    data_received = QtCore.pyqtSignal(np.ndarray, np.ndarray, float, float) # Raw, Clean, VAD, ProcTime

    def __init__(self):
        super().__init__()
        self.running = True
        self.rnnoise = RNNoiseWrapper(DLL_PATH)
        
        # PyAudio setup
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
        
        # Recording state
        self.is_recording = False
        self.wav_raw = None
        self.wav_clean = None
        
        print(f"Server started on {UDP_IP}:{UDP_PORT}")

    def start_recording(self):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        os.makedirs("recordings", exist_ok=True)
        
        self.wav_raw = wave.open(f"recordings/raw_{timestamp}.wav", 'wb')
        self.wav_raw.setnchannels(CHANNELS)
        self.wav_raw.setsampwidth(2)
        self.wav_raw.setframerate(SAMPLE_RATE)
        
        self.wav_clean = wave.open(f"recordings/clean_{timestamp}.wav", 'wb')
        self.wav_clean.setnchannels(CHANNELS)
        self.wav_clean.setsampwidth(2)
        self.wav_clean.setframerate(SAMPLE_RATE)
        
        self.is_recording = True
        print(f"Recording started: recordings/clean_{timestamp}.wav")

    def stop_recording(self):
        self.is_recording = False
        if self.wav_raw: self.wav_raw.close()
        if self.wav_clean: self.wav_clean.close()
        print("Recording stopped.")

    def run(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(2048)
                if len(data) == FRAME_SIZE * 2: # 16-bit PCM
                    t_start = time.perf_counter()
                    
                    # Convert to float32 for RNNoise
                    raw_audio = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                    
                    # Denoise
                    clean_audio, vad = self.rnnoise.process(raw_audio)
                    
                    # Measurement
                    t_end = time.perf_counter()
                    proc_time_ms = (t_end - t_start) * 1000
                    
                    # Convert back to int16 for playback
                    playback_data = clean_audio.clip(-32768, 32767).astype(np.int16)
                    self.stream.write(playback_data.tobytes())
                    
                    # Recording
                    if self.is_recording:
                        raw_bytes = np.frombuffer(data, dtype=np.int16).tobytes()
                        self.wav_raw.writeframes(raw_bytes)
                        self.wav_clean.writeframes(playback_data.tobytes())
                    
                    # Signal for GUI
                    self.data_received.emit(raw_audio, clean_audio, vad, proc_time_ms)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Error: {e}")
                break

    def stop(self):
        self.running = False
        self.stop_recording()
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        self.sock.close()

# --- GUI Dashboard ---
class Dashboard(QtWidgets.QMainWindow):
    def __init__(self, server):
        super().__init__()
        self.server = server
        self.setWindowTitle("ESP32-S3 RNNoise Advanced Lab Dashboard (C-Hybrid)")
        self.resize(1200, 800)
        
        # UI Styling
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; color: #E0E0E0; }
            QLabel { color: #E0E0E0; font-family: 'Segoe UI', sans-serif; }
            QPushButton { 
                background-color: #333; color: white; border-radius: 5px; padding: 10px;
                font-weight: bold; border: 1px solid #444;
            }
            QPushButton:checked { background-color: #D32F2F; border: 1px solid #FF5252; }
            QProgressBar { border: 1px solid #444; border-radius: 5px; text-align: center; color: white; background-color: #222; }
            QProgressBar::chunk { background-color: #4CAF50; }
        """)

        # Layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)
        
        # Left Panel (Controls & Stats)
        left_panel = QtWidgets.QVBoxLayout()
        main_layout.addLayout(left_panel, 1)

        # Header
        header = QtWidgets.QLabel("SYSTEM STATUS")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #4CAF50;")
        left_panel.addWidget(header)

        self.info_label = QtWidgets.QLabel("Status: Waiting for ESP32...")
        left_panel.addWidget(self.info_label)

        # Latency Monitor
        accel_box = QtWidgets.QGroupBox("Latency & Performance")
        accel_box.setStyleSheet("color: white; font-weight: bold;")
        accel_layout = QtWidgets.QVBoxLayout()
        self.latency_label = QtWidgets.QLabel("Proc Time: --- ms")
        self.latency_label.setStyleSheet("font-size: 24px; color: #FFD600;")
        accel_layout.addWidget(self.latency_label)
        self.load_label = QtWidgets.QLabel("CPU Load (Denoise): 0%")
        accel_layout.addWidget(self.load_label)
        accel_box.setLayout(accel_layout)
        left_panel.addWidget(accel_box)

        # VAD Meter
        left_panel.addWidget(QtWidgets.QLabel("Voice Activity Detection:"))
        self.vad_bar = QtWidgets.QProgressBar()
        self.vad_bar.setRange(0, 100)
        left_panel.addWidget(self.vad_bar)

        # Controls
        left_panel.addStretch()
        self.record_btn = QtWidgets.QPushButton("🔴 START RECORDING")
        self.record_btn.setCheckable(True)
        self.record_btn.clicked.connect(self.toggle_recording)
        left_panel.addWidget(self.record_btn)

        # Right Panel (Visualizers)
        right_panel = QtWidgets.QVBoxLayout()
        main_layout.addLayout(right_panel, 3)

        # Waveform Plot
        self.wave_plot = pg.PlotWidget(title="Waveform Comparison")
        self.wave_plot.setYRange(-15000, 15000)
        self.wave_plot.showGrid(x=True, y=True)
        self.wave_plot.addLegend()
        self.raw_curve = self.wave_plot.plot(pen='r', name="Raw (ESP32)")
        self.clean_curve = self.wave_plot.plot(pen='g', name="Cleaned (C-Backend)")
        right_panel.addWidget(self.wave_plot, 1)

        # Spectrogram Plot
        self.spec_plot = pg.PlotWidget(title="Frequency Spectrogram (Waterfall)")
        self.img = pg.ImageItem()
        self.spec_plot.addItem(self.img)
        
        # Color Map for Spectrogram
        pos = np.array([0., 0.2, 0.5, 0.8, 1.0])
        color = np.array([[0,0,0,255], [0,0,255,255], [0,255,0,255], [255,255,0,255], [255,0,0,255]], dtype=np.ubyte)
        cmap = pg.ColorMap(pos, color)
        self.img.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
        
        right_panel.addWidget(self.spec_plot, 1)

        # Buffers
        self.raw_buffer = np.zeros(FRAME_SIZE * 20)
        self.clean_buffer = np.zeros(FRAME_SIZE * 20)
        
        # Spectrogram Buffer (Scroll effect)
        self.n_fft = 512
        self.spec_history = 100
        self.spec_data = np.zeros((self.spec_history, self.n_fft // 2))
        
        # Connect signals
        self.server.data_received.connect(self.update_gui)
        self.last_update = time.time()

    def toggle_recording(self):
        if self.record_btn.isChecked():
            self.record_btn.setText("⏹ STOP RECORDING")
            self.server.start_recording()
        else:
            self.record_btn.setText("🔴 START RECORDING")
            self.server.stop_recording()

    def update_gui(self, raw, clean, vad, proc_time):
        # Update buffers
        self.raw_buffer = np.roll(self.raw_buffer, -len(raw))
        self.raw_buffer[-len(raw):] = raw
        
        self.clean_buffer = np.roll(self.clean_buffer, -len(clean))
        self.clean_buffer[-len(clean):] = clean
        
        # Update curves
        self.raw_curve.setData(self.raw_buffer)
        self.clean_curve.setData(self.clean_buffer)
        
        # Update VAD
        self.vad_bar.setValue(int(vad * 100))
        
        # Update Spectrogram (using clean audio)
        # Simple FFT
        window = np.hanning(len(clean))
        fft_data = np.abs(np.fft.fft(clean * window, n=self.n_fft))[:self.n_fft // 2]
        # Log scale for visibility
        fft_log = 20 * np.log10(fft_data + 1e-6)
        fft_norm = np.clip((fft_log + 60) / 100, 0, 1) # Normalize -60dB to 40dB range
        
        self.spec_data = np.roll(self.spec_data, -1, axis=0)
        self.spec_data[-1, :] = fft_norm
        self.img.setImage(self.spec_data.T)
        
        # Update status & latency
        self.latency_label.setText(f"Proc Time: {proc_time:.3f} ms")
        load = (proc_time / 10.0) * 100 # 10ms frame budget
        self.load_label.setText(f"Inference Load: {load:.1f}%")

        now = time.time()
        if now - self.last_update > 1.0:
            self.info_label.setText(f"Connected | VAD: {vad:.2f} | C-Hybrid Mode Active")
            self.last_update = now

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    
    server = AudioServer()
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    
    gui = Dashboard(server)
    gui.show()
    
    def on_exit():
        server.stop()
        app.quit()
        
    app.aboutToQuit.connect(on_exit)
    sys.exit(app.exec_())
