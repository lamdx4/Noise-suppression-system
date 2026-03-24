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
        
        # Runtime params
        self.bypass = False
        self.digital_gain = 1.0
        
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

    def stop_recording(self):
        self.is_recording = False
        if self.wav_raw: self.wav_raw.close()
        if self.wav_clean: self.wav_clean.close()

    def run(self):
        packet_count = 0
        while self.running:
            try:
                data, addr = self.sock.recvfrom(2048)
                if packet_count < 10:
                    print(f"Received packet {packet_count}: {len(data)} bytes from {addr}")
                    packet_count += 1
                
                if len(data) == FRAME_SIZE * 2: # 16-bit PCM
                    t_start = time.perf_counter()
                    raw_audio = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                    
                    if not self.bypass:
                        clean_audio, vad = self.rnnoise.process(raw_audio)
                    else:
                        clean_audio = raw_audio.copy()
                        vad = 0.0
                    
                    # Apply Digital Gain
                    clean_audio *= self.digital_gain
                    
                    t_end = time.perf_counter()
                    proc_time_ms = (t_end - t_start) * 1000
                    
                    # Playback
                    playback_data = clean_audio.clip(-32768, 32767).astype(np.int16)
                    self.stream.write(playback_data.tobytes())
                    
                    # Recording
                    if self.is_recording:
                        self.wav_raw.writeframes(data)
                        self.wav_clean.writeframes(playback_data.tobytes())
                    
                    self.data_received.emit(raw_audio, clean_audio, vad, proc_time_ms)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
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
        self.resize(1300, 850)
        
        self.setStyleSheet("""
            QMainWindow { background-color: #0F0F0F; color: #EEE; }
            QLabel { color: #EEE; font-family: 'Consolas', monospace; }
            QGroupBox { border: 2px solid #333; border-radius: 8px; margin-top: 10px; font-weight: bold; padding: 10px; }
            QPushButton { 
                background-color: #222; color: white; border-radius: 6px; padding: 12px;
                font-weight: bold; border: 1px solid #444; min-width: 150px;
            }
            QPushButton:hover { background-color: #333; }
            QPushButton:checked { background-color: #B71C1C; }
            QProgressBar { height: 15px; border: 1px solid #444; border-radius: 7px; text-align: center; background: #111; }
            QProgressBar::chunk { background-color: #00E676; }
            QSlider::handle:horizontal { background: #4CAF50; border-radius: 5px; width: 18px; }
        """)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)
        
        # Left Panel (Controls)
        left_panel = QtWidgets.QVBoxLayout()
        main_layout.addLayout(left_panel, 1)

        # System Header
        title = QtWidgets.QLabel("RNNOISE ENGINE v2.0")
        title.setStyleSheet("font-size: 20px; color: #00E676; font-weight: bold;")
        left_panel.addWidget(title)

        # Performance Box
        perf_box = QtWidgets.QGroupBox("PERFORMANCE MONITOR")
        perf_layout = QtWidgets.QVBoxLayout()
        self.latency_label = QtWidgets.QLabel("Inference: --- ms")
        self.latency_label.setStyleSheet("font-size: 22px; color: #FFEA00;")
        perf_layout.addWidget(self.latency_label)
        self.load_label = QtWidgets.QLabel("Load: 0%")
        perf_layout.addWidget(self.load_label)
        perf_box.setLayout(perf_layout)
        left_panel.addWidget(perf_box)

        # Audio Processing Box
        proc_box = QtWidgets.QGroupBox("PROCESSING CONTROLS")
        proc_layout = QtWidgets.QVBoxLayout()
        
        self.bypass_btn = QtWidgets.QPushButton("BYPASS (OFF)")
        self.bypass_btn.setCheckable(True)
        self.bypass_btn.clicked.connect(self.toggle_bypass)
        proc_layout.addWidget(self.bypass_btn)
        
        proc_layout.addWidget(QtWidgets.QLabel("Digital Gain:"))
        self.gain_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gain_slider.setRange(0, 400) # 0.0 to 4.0
        self.gain_slider.setValue(100)
        self.gain_slider.valueChanged.connect(self.update_gain)
        proc_layout.addWidget(self.gain_slider)
        self.gain_label = QtWidgets.QLabel("Gain: 1.0x")
        proc_layout.addWidget(self.gain_label)
        
        proc_box.setLayout(proc_layout)
        left_panel.addWidget(proc_box)

        # VAD Section
        left_panel.addWidget(QtWidgets.QLabel("VOICE ACTIVITY:"))
        self.vad_bar = QtWidgets.QProgressBar()
        left_panel.addWidget(self.vad_bar)

        left_panel.addStretch()
        self.record_btn = QtWidgets.QPushButton("🔴 START RECORDING")
        self.record_btn.setCheckable(True)
        self.record_btn.clicked.connect(self.toggle_recording)
        left_panel.addWidget(self.record_btn)

        # Right Panel (Visuals)
        right_panel = QtWidgets.QVBoxLayout()
        main_layout.addLayout(right_panel, 3)

        self.wave_plot = pg.PlotWidget(title="Live Waveform (RED:Raw, GREEN:Clean)")
        self.wave_plot.setLabel('left', 'Amplitude', units='PCM')
        self.wave_plot.setLabel('bottom', 'Samples')
        self.wave_plot.setYRange(-16000, 16000)
        self.wave_plot.showGrid(x=True, y=True)
        self.raw_curve = self.wave_plot.plot(pen='r')
        self.clean_curve = self.wave_plot.plot(pen=pg.mkPen('#00E676', width=1.5))
        right_panel.addWidget(self.wave_plot, 1)

        self.freq_plot = pg.PlotWidget(title="Frequency Spectrum (Amplitude vs Frequency)")
        self.freq_plot.setLabel('left', 'Magnitude', units='dB')
        self.freq_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.freq_plot.setYRange(-60, 40)
        self.freq_plot.setXRange(0, SAMPLE_RATE // 2)
        self.freq_plot.showGrid(x=True, y=True)
        self.raw_freq_curve = self.freq_plot.plot(pen=pg.mkPen('r', width=1, style=QtCore.Qt.DotLine), name="Raw")
        self.clean_freq_curve = self.freq_plot.plot(pen=pg.mkPen('#00E676', width=1.5), name="Clean")
        right_panel.addWidget(self.freq_plot, 1)

        # --- Spectrogram (Waterfall) ---
        self.spectro_plot = pg.PlotWidget(title="Real-time Spectrogram (Waterfall)")
        self.spectro_plot.setLabel('left', 'Frequency', units='Hz')
        self.spectro_plot.setLabel('bottom', 'Time')
        self.spectro_plot.setXRange(0, 100) # 100 frames
        self.spectro_plot.setYRange(0, SAMPLE_RATE // 2)
        
        self.img = pg.ImageItem()
        self.spectro_plot.addItem(self.img)
        
        # Color Map
        colormap = pg.colormap.get('viridis')
        bar = pg.ColorBarItem(values=(-60, 20), colorMap=colormap)
        bar.setImageItem(self.img)
        right_panel.addWidget(self.spectro_plot, 1)

        # --- Clean Spectrogram (Waterfall) ---
        self.spectro_plot_clean = pg.PlotWidget(title="Clean Spectrogram (Waterfall)")
        self.spectro_plot_clean.setLabel('left', 'Frequency', units='Hz')
        self.spectro_plot_clean.setLabel('bottom', 'Time')
        self.spectro_plot_clean.setXRange(0, 100)
        self.spectro_plot_clean.setYRange(0, SAMPLE_RATE // 2)
        
        self.img_clean = pg.ImageItem()
        self.spectro_plot_clean.addItem(self.img_clean)
        
        bar_clean = pg.ColorBarItem(values=(-60, 20), colorMap=colormap)
        bar_clean.setImageItem(self.img_clean)
        right_panel.addWidget(self.spectro_plot_clean, 1)

        # Setup Buffers
        self.raw_buffer = np.zeros(FRAME_SIZE * 20)
        self.clean_buffer = np.zeros(FRAME_SIZE * 20)
        self.n_fft = 512
        self.num_rows = 100 # History depth
        self.spectro_data = np.full((self.num_rows, self.n_fft // 2 + 1), -60.0)
        self.spectro_data_clean = np.full((self.num_rows, self.n_fft // 2 + 1), -60.0)
        
        # Scale images to match axes
        rect = QtCore.QRectF(0, 0, self.num_rows, SAMPLE_RATE // 2)
        self.img.setRect(rect)
        self.img_clean.setRect(rect)
        
        self.freq_axis = np.fft.rfftfreq(self.n_fft, 1/SAMPLE_RATE)
        self.window = np.hanning(FRAME_SIZE) # Pre-calculate window
        
        # Throttling
        self.update_counter = 0
        
        self.server.data_received.connect(self.update_gui)

    def toggle_bypass(self):
        self.server.bypass = self.bypass_btn.isChecked()
        self.bypass_btn.setText("BYPASS (ON)" if self.server.bypass else "BYPASS (OFF)")
        self.bypass_btn.setStyleSheet("background-color: #FF5722;" if self.server.bypass else "")

    def update_gain(self):
        val = self.gain_slider.value() / 100.0
        self.server.digital_gain = val
        self.gain_label.setText(f"Gain: {val:.2f}x")

    def toggle_recording(self):
        if self.record_btn.isChecked():
            self.record_btn.setText("⏹ STOP RECORDING")
            self.server.start_recording()
        else:
            self.record_btn.setText("🔴 START RECORDING")
            self.server.stop_recording()

    def update_gui(self, raw, clean, vad, proc_time):
        # Always update buffers to keep data continuous
        self.raw_buffer = np.roll(self.raw_buffer, -len(raw))
        self.raw_buffer[-len(raw):] = raw
        self.clean_buffer = np.roll(self.clean_buffer, -len(clean))
        self.clean_buffer[-len(clean):] = clean
        
        # Update Spectrogram Buffers (Calculated every frame for smoothness)
        raw_fft = np.abs(np.fft.rfft(raw * self.window, n=self.n_fft))
        raw_db = 20 * np.log10(raw_fft + 1e-6)
        
        clean_fft = np.abs(np.fft.rfft(clean * self.window, n=self.n_fft))
        clean_db = 20 * np.log10(clean_fft + 1e-6)
        
        self.spectro_data = np.roll(self.spectro_data, -1, axis=0)
        self.spectro_data[-1, :] = raw_db
        
        self.spectro_data_clean = np.roll(self.spectro_data_clean, -1, axis=0)
        self.spectro_data_clean[-1, :] = clean_db

        # Only update the visual plots every 3 frames (~33 FPS) to avoid lag
        self.update_counter += 1
        if self.update_counter % 3 != 0:
            return

        self.raw_curve.setData(self.raw_buffer)
        self.clean_curve.setData(self.clean_buffer)
        self.vad_bar.setValue(int(vad * 100))
        
        self.raw_freq_curve.setData(self.freq_axis, raw_db)
        self.clean_freq_curve.setData(self.freq_axis, clean_db)
        
        # Update Waterfall Images
        self.img.setImage(self.spectro_data.T, autoLevels=False)
        self.img_clean.setImage(self.spectro_data_clean.T, autoLevels=False)
        
        self.latency_label.setText(f"Proc: {proc_time:.3f} ms")
        self.load_label.setText(f"Load: {(proc_time / 10.0) * 100:.1f}%")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    server = AudioServer()
    threading.Thread(target=server.run, daemon=True).start()
    gui = Dashboard(server)
    gui.show()
    app.aboutToQuit.connect(server.stop)
    sys.exit(app.exec_())
