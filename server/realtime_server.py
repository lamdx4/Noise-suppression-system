import sys
import socket
import threading
import numpy as np
import pyaudio
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets
import ctypes
import time

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
    data_received = QtCore.pyqtSignal(np.ndarray, np.ndarray, float) # Raw, Clean, VAD

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
        
        print(f"Server started on {UDP_IP}:{UDP_PORT}")

    def run(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(2048)
                if len(data) == FRAME_SIZE * 2: # 16-bit PCM
                    # Convert to float32 for RNNoise
                    raw_audio = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                    
                    # Denoise
                    clean_audio, vad = self.rnnoise.process(raw_audio)
                    
                    # Convert back to int16 for playback
                    playback_data = clean_audio.clip(-32768, 32767).astype(np.int16)
                    self.stream.write(playback_data.tobytes())
                    
                    # Signal for GUI
                    self.data_received.emit(raw_audio, clean_audio, vad)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Error: {e}")
                break

    def stop(self):
        self.running = False
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        self.sock.close()

# --- GUI Dashboard ---
class Dashboard(QtWidgets.QMainWindow):
    def __init__(self, server):
        super().__init__()
        self.server = server
        self.setWindowTitle("ESP32-S3 RNNoise Real-time Dashboard (C-Hybrid)")
        self.resize(1000, 600)
        
        # Layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Info Label
        self.info_label = QtWidgets.QLabel("Waiting for ESP32...")
        self.info_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #4CAF50;")
        layout.addWidget(self.info_label)
        
        # Plot (Waveform)
        self.plot_widget = pg.PlotWidget(title="Real-time Audio Waveform")
        self.plot_widget.setYRange(-10000, 10000)
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.addLegend()
        
        self.raw_curve = self.plot_widget.plot(pen='r', name="Raw (ESP32)")
        self.clean_curve = self.plot_widget.plot(pen='g', name="Cleaned (C-Backend)")
        layout.addWidget(self.plot_widget)
        
        # VAD Meter
        self.vad_bar = QtWidgets.QProgressBar()
        self.vad_bar.setRange(0, 100)
        self.vad_bar.setFormat("Voice Activity: %p%")
        layout.addWidget(QtWidgets.QLabel("VAD Probability:"))
        layout.addWidget(self.vad_bar)
        
        # Buffers for plotting
        self.raw_buffer = np.zeros(FRAME_SIZE * 10)
        self.clean_buffer = np.zeros(FRAME_SIZE * 10)
        
        # Connect signals
        self.server.data_received.connect(self.update_gui)
        
        # Status update timer
        self.last_update = time.time()

    def update_gui(self, raw, clean, vad):
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
        
        # Update status
        now = time.time()
        if now - self.last_update > 1.0:
            self.info_label.setText(f"Connected | VAD: {vad:.2f} | Latency: Low (C-Backend)")
            self.last_update = now

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    
    # App style
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
