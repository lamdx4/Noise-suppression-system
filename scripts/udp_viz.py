import socket
import struct
import pyaudio
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import threading
import queue
import time
import sys

# === CONFIGURATION ===
UDP_IP = "0.0.0.0"
UDP_PORT = 12345
SAMPLE_RATE = 48000
FRAME_SIZE = 480          # Size of one UDP packet (samples)
FFT_WINDOW_SIZE = 2048    # Analysis window for FFT
UI_REFRESH_RATE_MS = 33   # ~30 FPS

class AudioVisualizer:
    def __init__(self):
        # 1. Setup Queues
        # Audio Queue: Large buffer to prevent stuttering (latency trade-off)
        self.audio_queue = queue.Queue(maxsize=50) 
        # Viz Queue: Small buffer, drop old frames if UI is slow
        self.viz_queue = queue.Queue(maxsize=50)
        
        # 2. Setup FFT Buffer
        self.rolling_buffer = np.zeros(FFT_WINDOW_SIZE, dtype=np.float32)
        self.window_func = np.hanning(FFT_WINDOW_SIZE)
        
        # 3. Setup GUI
        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(title="Real-time Audio Spectrum")
        self.win.resize(1000, 600)
        
        # Plot
        self.plot = self.win.addPlot(title="Frequency Spectrum (RNNoise Output)")
        self.plot.setLabel('left', "Magnitude", units='dB')
        self.plot.setLabel('bottom', "Frequency", units='Hz')
        self.plot.setLogMode(x=True, y=False)
        self.plot.setYRange(-60, 100)
        self.plot.setXRange(np.log10(20), np.log10(SAMPLE_RATE/2))
        self.plot.showGrid(x=True, y=True)
        self.curve = self.plot.plot(pen=pg.mkPen('c', width=2))
        
        self.win.show()
        
        # 4. Setup Audio (Blocking Mode - Like udp_player.py)
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            output=True,
            frames_per_buffer=FRAME_SIZE
        )
        
        self.running = True
        
        # 5. Start Threads
        # Thread 1: UDP Receiver
        self.udp_thread = threading.Thread(target=self.udp_receiver)
        self.udp_thread.daemon = True
        self.udp_thread.start()
        
        # Thread 2: Audio Player (Dedicated)
        self.audio_thread = threading.Thread(target=self.audio_player)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        
        # 6. Start UI Timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(UI_REFRESH_RATE_MS)

    def udp_receiver(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536) # Increase UDP buffer
        sock.bind((UDP_IP, UDP_PORT))
        print(f"Listening on {UDP_IP}:{UDP_PORT}...")
        
        while self.running:
            try:
                data, _ = sock.recvfrom(4096) # Receive whatever is there
                
                # Check frame size
                if len(data) == 0: continue

                # Put to Audio Queue (Blocking if full to sync flow)
                self.audio_queue.put(data)
                
                # Put to Viz Queue (Non-blocking, drop if full)
                if not self.viz_queue.full():
                    self.viz_queue.put(data)
                
            except Exception as e:
                print(f"UDP Error: {e}")

    def audio_player(self):
        print("Audio thread started...")
        while self.running:
            try:
                # Get data (Blocking wait)
                data = self.audio_queue.get()
                # Write to stream (Blocking playback)
                self.stream.write(data)
            except Exception as e:
                print(f"Audio Error: {e}")

    def update_plot(self):
        # Fetch ALL available chunks for visualization
        chunks = []
        try:
            while True:
                chunks.append(self.viz_queue.get_nowait())
        except queue.Empty:
            pass
            
        if not chunks: return

        # Process latest chunks
        # Parse bytes -> Int16 -> Float32
        data_bytes = b''.join(chunks)
        audio_data = np.frombuffer(data_bytes, dtype=np.int16)
        
        # Update rolling buffer
        num_new = len(audio_data)
        if num_new >= FFT_WINDOW_SIZE:
             self.rolling_buffer[:] = audio_data[-FFT_WINDOW_SIZE:]
        else:
             self.rolling_buffer = np.roll(self.rolling_buffer, -num_new)
             self.rolling_buffer[-num_new:] = audio_data
             
        # FFT
        windowed = self.rolling_buffer * self.window_func
        spectrum = np.fft.rfft(windowed)
        magnitude = np.abs(spectrum)
        magnitude_db = 20 * np.log10(magnitude + 1e-6)
        
        freqs = np.linspace(0, SAMPLE_RATE/2, len(magnitude_db))
        self.curve.setData(freqs, magnitude_db)

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtWidgets.QApplication.instance().exec_()

if __name__ == "__main__":
    viz = AudioVisualizer()
    viz.start()
