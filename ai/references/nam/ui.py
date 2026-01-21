import os
import platform
import sys
import eel


is_ready = False


@eel.expose
def on_loaded():
    print("UI khởi động hoàn tất")
    global is_ready
    is_ready = True


def on_close(page, sockets_still_open):
    global is_ready
    is_ready = False
    print("Đóng UI")


def send_bytes_raw(data):
    global is_ready
    if not is_ready:
        return
    """
    Gửi dữ liệu nhị phân từ Python đến JavaScript
    :param data: Dữ liệu nhị phân
    """
    eel.receive_bytes_raw(data)


def send_bytes(data):
    global is_ready
    if not is_ready:
        return
    """
    Gửi dữ liệu nhị phân từ Python đến JavaScript
    :param data: Dữ liệu nhị phân
    """
    eel.receive_bytes(data)


def run():
    if not os.path.exists("ui/dist"):
        os.system("cd ui && yarn build && cd ..")

    eel.init("ui/dist")

    try:
        eel.start("index.html", size=(1400, 900), close_callback=on_close, block=True)
    except EnvironmentError:
        if sys.platform in ["win32", "win64"] and int(platform.release()) >= 10:
            eel.start(
                "index.html",
                mode="edge",
                size=(1400, 900),
                close_callback=on_close,
                block=True,
            )
        else:
            raise
