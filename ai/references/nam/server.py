import socket
import wave
import queue
import threading
import numpy as np
import pyaudio

from references.nam.model import unet_3blocks
from references.nam.processdata import prepare_data, istft
import references.nam.ui

# Server settings
HOST = '0.0.0.0'  # Lắng nghe trên tất cả các interface
PORT = 12345  # Cổng server
SAMPLE_RATE = 8000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit audio
BUFFER_SIZE = 500
SAMPLES_PER_SECOND = 8000
BUFFERS_PER_SECOND = SAMPLES_PER_SECOND // BUFFER_SIZE  # 8000 / 500 = 16

# Queues for raw and processed audio
raw_audio_queue = queue.Queue(maxsize=3)  # Chỉ chứa 1 block 8000 mẫu
processed_audio_queue = queue.Queue(maxsize=3)  # Chỉ chứa 1 block 8000 mẫu

# Initialize WAV file
wav_file = wave.open("output_processed.wav", "wb")
wav_file.setnchannels(CHANNELS)
wav_file.setsampwidth(SAMPLE_WIDTH)
wav_file.setframerate(SAMPLE_RATE)

# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                output=True)

checkpoint_file = "checkpoints/model_epoch_41.weights.h5"
model = unet_3blocks()
model.load_weights(checkpoint_file)

def main0():
    ui.run()
# def process_audio():
#     """Process audio data from raw queue and put into processed queue"""
#     while True:
#         try:
#             # Get raw audio data from queue
#             raw_data = raw_audio_queue.get(timeout=1)
#
#             # Convert to numpy array for processing
#             samples = np.frombuffer(raw_data, dtype=np.int16)
#             print(samples)
#             # Example processing: Amplify audio by 1.5x
#             processed_samples = (samples * 1.5).clip(-32768, 32767).astype(np.int16)
#             print(f"Processed samples: {processed_samples[:10]}...")  # Print first 10 samples for debugging
#             # Put processed data into processed queue
#             processed_audio_queue.put(processed_samples.tobytes())
#
#             raw_audio_queue.task_done()
#         except queue.Empty:
#             continue
#         except Exception as e:
#             print(f"Processing error: {e}")
#             break
def process_audio():
    """Process audio data from raw queue and put into processed queue"""
    while True:
        try:
            # Get raw audio data from queue
            raw_data = raw_audio_queue.get(timeout=1)

            # Convert to numpy array for processing
            samples = np.frombuffer(raw_data, dtype=np.int16)


            # normalize samples
            samples = samples/32768.0  # Normalize to [-1, 1]
            ui.send_bytes_raw(samples.tolist())
            data = prepare_data(samples, sample_rate=SAMPLE_RATE)
            output = model(data,training=False)
            res = istft(output)

            ui.send_bytes(res[0].tolist())
            res = res * 32768.0  # Scale back to original range
            res = np.clip(res, -32768, 32767).astype(np.int16)  # Clip to int16 range

            # Put processed data into processed queue
            processed_audio_queue.put(res.tobytes())

            raw_audio_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Processing error: {e}")
            break


def play_and_save_audio():
    """Play audio from processed queue, save to WAV, and remove from queue"""
    while True:
        try:
            # Get processed audio data from queue
            processed_data = processed_audio_queue.get(timeout=1)

            # Play audio
            stream.write(processed_data)

            # Save to WAV file
            wav_file.writeframes(processed_data)

            # Remove from queue after playing and saving
            processed_audio_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Play/Save error: {e}")
            break


def receive_audio(conn):
    """Receive audio data from ESP32 and put into raw queue when 8000 samples are collected"""
    temp_buffer = bytearray()
    while True:
        try:
            data = conn.recv(BUFFER_SIZE * SAMPLE_WIDTH)
            if not data:
                break
            temp_buffer.extend(data)

            # Check if we have enough samples (8000 samples = 16 buffers of 500 samples)
            if len(temp_buffer) >= SAMPLES_PER_SECOND * SAMPLE_WIDTH:
                # Take exactly 8000 samples
                raw_audio_queue.put(temp_buffer[:SAMPLES_PER_SECOND * SAMPLE_WIDTH])

                # Remove used data from temp_buffer
                temp_buffer = temp_buffer[SAMPLES_PER_SECOND * SAMPLE_WIDTH:]
        except Exception as e:
            print(f"Receive error: {e}")
            break


def main():
    ui_thread = threading.Thread(target=main0, args=())
    ui_thread.start()
    # Create socket server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)

    print(f"Server listening on {HOST}:{PORT}")
    conn, addr = server_socket.accept()
    print(f"Connected by {addr}")

    # Start processing and playing/saving threads
    processing_thread = threading.Thread(target=process_audio, daemon=True)
    playing_saving_thread = threading.Thread(target=play_and_save_audio, daemon=True)


    processing_thread.start()
    playing_saving_thread.start()

    # Receive audio data
    receive_audio(conn)

    # Cleanup
    stream.stop_stream()
    stream.close()
    p.terminate()
    conn.close()
    server_socket.close()
    wav_file.close()
    print("Server stopped")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Server stopped by user")