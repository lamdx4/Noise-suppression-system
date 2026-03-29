import sys
from PyQt5 import QtCore, QtWidgets
from audio_worker import AudioServer
from dashboard_ui import Dashboard

"""
RNNoise Advanced Lab Dashboard v2.0 (Component Architecture)
- Lõi Worker: QThread + AudioServer (Background)
- UI: Pyqtgraph GraphicsLayout (Stable Rendering)
- Math: Numpy Fast FFT (Low Overhead)
"""

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion") # Dark/White hybrid look
    
    # 1. Khởi tạo Worker & Thread
    # Chú ý: Không truyền tham số 'parent' cho AudioServer để dùng moveToThread
    server = AudioServer()
    thread = QtCore.QThread()
    
    # 2. Chuyển Worker sang Thread riêng biệt
    server.moveToThread(thread)
    
    # 3. Cấu hình vòng đời
    # Khi thread bắt đầu, hàm run sẽ tự động kích hoạt
    thread.started.connect(server.run)
    
    # Đảm bảo dọn dẹp tài nguyên khi tắt ứng dụng
    print("Connecting cleanup signals...")
    app.aboutToQuit.connect(server.stop)
    app.aboutToQuit.connect(thread.quit)
    app.aboutToQuit.connect(thread.wait)

    
    # 4. Khởi tạo Giao diện (Truyền server vào để UI lắng nghe Signal)
    gui = Dashboard(server)
    gui.show()
    
    # 5. Kích hoạt Thread và Vòng lặp sự kiện chính
    thread.start()
    
    print("--- Dashboard System Online (30 FPS Throttled) ---")
    sys.exit(app.exec_())
