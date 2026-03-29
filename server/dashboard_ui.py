import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg

# Link themes to GraphicsLayoutWidget
pg.setConfigOption('background', '#FFFFFF')
pg.setConfigOption('foreground', '#333333')

class Dashboard(QtWidgets.QMainWindow):
    def __init__(self, server):
        super().__init__()
        self.server = server
        self.setWindowTitle("ESP32-S3 RNNoise Advanced Lab Dashboard (Component-Alpha)")
        self.resize(1200, 800)
        
        self.setStyleSheet("""
            QMainWindow { background-color: #FFFFFF; color: #333333; }
            QLabel { color: #333333; font-family: 'Consolas', monospace; }
            QGroupBox { border: 2px solid #DDDDDD; border-radius: 8px; margin-top: 10px; font-weight: bold; padding: 10px; }
            QPushButton { background-color: #F0F0F0; border-radius: 6px; padding: 10px; font-weight: bold; border: 1px solid #CCCCCC; }
            QPushButton:checked { background-color: #FFCDD2; color: #B71C1C; }
            QProgressBar { height: 12px; border: 1px solid #CCCCCC; border-radius: 6px; text-align: center; background: #F9F9F9; }
            QProgressBar::chunk { background-color: #4CAF50; border-radius: 5px; }
        """)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)
        
        # Left Panel (Controls)
        left_panel = QtWidgets.QVBoxLayout()
        main_layout.addLayout(left_panel, 1)

        header = QtWidgets.QLabel("RNNOISE v2.0 (STABLE)")
        header.setStyleSheet("font-size: 18px; color: #2E7D32; font-weight: bold;")
        left_panel.addWidget(header)

        # Performance Monitor
        perf_box = QtWidgets.QGroupBox("PERFORMANCE")
        perf_layout = QtWidgets.QVBoxLayout()
        self.latency_label = QtWidgets.QLabel("Inference: --- ms")
        self.latency_label.setStyleSheet("font-size: 20px; color: #D32F2F;")
        perf_layout.addWidget(self.latency_label)
        self.load_label = QtWidgets.QLabel("CPU Load: 0%")
        perf_layout.addWidget(self.load_label)
        perf_box.setLayout(perf_layout)
        left_panel.addWidget(perf_box)

        # Controls
        ctrl_box = QtWidgets.QGroupBox("CONTROLS")
        ctrl_layout = QtWidgets.QVBoxLayout()
        self.bypass_btn = QtWidgets.QPushButton("BYPASS (OFF)")
        self.bypass_btn.setCheckable(True)
        self.bypass_btn.clicked.connect(self.toggle_bypass)
        ctrl_layout.addWidget(self.bypass_btn)
        
        ctrl_layout.addWidget(QtWidgets.QLabel("Digital Gain:"))
        self.gain_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gain_slider.setRange(0, 400); self.gain_slider.setValue(100)
        self.gain_slider.valueChanged.connect(self.update_gain)
        ctrl_layout.addWidget(self.gain_slider)
        self.gain_label = QtWidgets.QLabel("Gain: 1.0x")
        ctrl_layout.addWidget(self.gain_label)
        ctrl_box.setLayout(ctrl_layout)
        left_panel.addWidget(ctrl_box)

        # VAD
        left_panel.addWidget(QtWidgets.QLabel("VOICE ACTIVITY:"))
        self.vad_bar = QtWidgets.QProgressBar()
        left_panel.addWidget(self.vad_bar)
        
        left_panel.addStretch()
        self.record_btn = QtWidgets.QPushButton("🔴 START RECORDING")
        self.record_btn.setCheckable(True); self.record_btn.clicked.connect(self.toggle_recording)
        left_panel.addWidget(self.record_btn)

        # --- Right Panel: GraphicsLayoutWidget ---
        self.view_widget = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.view_widget, 3)
        
        # 1. Oscilloscope
        self.p1 = self.view_widget.addPlot(title="Waveform (Normalized)")
        self.p1.setYRange(-1.1, 1.1); self.p1.showGrid(x=True, y=True)
        self.p1.setLabel('left', 'Amp')
        self.raw_curve = self.p1.plot(pen=pg.mkPen('#FF1744', width=2.2)) # Đỏ rực sắc nét
        self.clean_curve = self.p1.plot(pen=pg.mkPen('#00C853', width=2.2)) # Xanh lục đậm hơn
        
        self.view_widget.nextRow()
        
        # 2. Spectrum 1D
        self.p2 = self.view_widget.addPlot(title="Spectrum (dB Magnitude)")
        self.p2.setYRange(-80, 20); self.p2.showGrid(x=True, y=True)
        self.p2.setLabel('left', 'dB'); self.p2.setLabel('bottom', 'Hz')
        self.raw_f_curve = self.p2.plot(pen=pg.mkPen('#FF1744', width=1.5, style=QtCore.Qt.DotLine))
        self.clean_f_curve = self.p2.plot(pen=pg.mkPen('#00C853', width=2.2))
        
        self.view_widget.nextRow()
        
        # 3. Waterfall (Spectrograms)
        # Tạo sub-layout cho 2 biểu đồ ngang
        waterfall_box = self.view_widget.addLayout()
        
        # Raw Waterfall
        p3 = waterfall_box.addPlot(title="Waterfall: RAW")
        p3.setYRange(0, 24000)
        self.img_raw = pg.ImageItem()
        p3.addItem(self.img_raw)
        
        colormap = pg.colormap.get('inferno')
        bar1 = pg.ColorBarItem(colorMap=colormap)
        bar1.setImageItem(self.img_raw); bar1.setLevels(low=-80, high=0)
        waterfall_box.addItem(bar1)
        
        # Clean Waterfall
        p4 = waterfall_box.addPlot(title="Waterfall: CLEAN")
        p4.setYRange(0, 24000); p4.setXLink(p3); p4.setYLink(p3)
        self.img_clean = pg.ImageItem()
        p4.addItem(self.img_clean)
        
        bar2 = pg.ColorBarItem(colorMap=colormap)
        bar2.setImageItem(self.img_clean); bar2.setLevels(low=-80, high=0)
        waterfall_box.addItem(bar2)

        # Data Management
        self.num_rows = 100; self.n_bins = 257 # for n_fft=512
        self.spec_raw = np.full((self.num_rows, self.n_bins), -100.0)
        self.spec_clean = np.full((self.num_rows, self.n_bins), -100.0)
        
        # Rect mapping (100 frames over 1s, 24kHz)
        rect = QtCore.QRectF(0, 0, 1.0, 24000)
        self.img_raw.setRect(rect)
        self.img_clean.setRect(rect)
        self.freq_axis = np.fft.rfftfreq(512, 1/48000)

        # Connection
        server.ui_update.connect(self.handle_ui_update)

    def toggle_bypass(self):
        self.server.bypass = self.bypass_btn.isChecked()
        self.bypass_btn.setText("BYPASS (ON)" if self.server.bypass else "BYPASS (OFF)")

    def update_gain(self):
        val = self.gain_slider.value() / 100.0
        self.server.digital_gain = val
        self.gain_label.setText(f"Gain: {val:.2f}x")

    def toggle_recording(self):
        if self.record_btn.isChecked():
            self.record_btn.setText("⏹ STOP"); self.server.start_recording()
        else:
            self.record_btn.setText("🔴 START RECORDING"); self.server.stop_recording()

    def handle_ui_update(self, data):
        # Update Plots

        self.raw_curve.setData(data['wave_raw'])
        self.clean_curve.setData(data['wave_clean'])
        self.raw_f_curve.setData(self.freq_axis, data['db_raw'])
        self.clean_f_curve.setData(self.freq_axis, data['db_clean'])
        self.vad_bar.setValue(int(data['vad'] * 100))
        self.latency_label.setText(f"Inf: {data['proc_time']:.2f}ms")
        self.load_label.setText(f"Load: {(data['proc_time']/10)*100:.1f}%")
        self.spec_raw = np.roll(self.spec_raw, -1, axis=0)
        self.spec_raw[-1, :] = data['db_raw']
        self.spec_clean = np.roll(self.spec_clean, -1, axis=0)
        self.spec_clean[-1, :] = data['db_clean']
        self.img_raw.setImage(self.spec_raw, autoLevels=False, levels=(-80, 0))
        self.img_clean.setImage(self.spec_clean, autoLevels=False, levels=(-80, 0))

    def closeEvent(self, event):
        """Khi bấm nút đóng cửa sổ, báo cho App thoát sạch sẽ"""
        print("Closing Dashboard...")
        self.server.stop()
        QtCore.QCoreApplication.quit()
        event.accept()
