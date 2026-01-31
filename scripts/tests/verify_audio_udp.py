
import socket
import wave
import time
import os

# Configuration
UDP_IP = "0.0.0.0"  # Listen on all interfaces
UDP_PORT = 12345    # Must match ESP32 target port
DURATION = 10       # Target recording duration in seconds
OUTPUT_FILE = "recorded_rnnoise_output.wav"

# Audio Format (Standard RNNoise / ESP32 I2S settings)
SAMPLE_RATE = 48000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit = 2 bytes

def main():
    expected_bytes = SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH * DURATION
    
    print(f"--- UDP Audio Recorder Tool (Byte-Exact Mode) ---")
    print(f"Listening on {UDP_IP}:{UDP_PORT}")
    print(f"Format: {SAMPLE_RATE}Hz, {CHANNELS} Channel, 16-bit PCM")
    print(f"Target Duration: {DURATION} seconds ({expected_bytes} bytes)")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Increase socket buffer size to prevent dropping packets
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024) 
    
    try:
        sock.bind((UDP_IP, UDP_PORT))
    except Exception as e:
        print(f"Error binding to port: {e}")
        return

    frames = []
    
    print("\n[Waiting for first packet...]")
    
    # Wait for first packet to start session
    sock.settimeout(None) # Wait indefinitely for the first packet
    data, addr = sock.recvfrom(4096)
    frames.append(data)
    print(f"[Started] Receiving from {addr}")
    
    start_time = time.time()
    packet_count = 1
    total_bytes = len(data)

    while total_bytes < expected_bytes:
        try:
            # Short timeout to detect if streams stops completely
            sock.settimeout(2.0) 
            data, _ = sock.recvfrom(4096)
            frames.append(data)
            packet_count += 1
            total_bytes += len(data)
            
            # Progress indicator
            if packet_count % 20 == 0:
                percent = (total_bytes / expected_bytes) * 100
                elapsed = time.time() - start_time
                print(f"\rProgress: {percent:.1f}% | Bytes: {total_bytes}/{expected_bytes} | Time: {elapsed:.1f}s", end="")
        
        except socket.timeout:
            print("\n[Timeout] No data for 2 seconds. Ending early.")
            break
        except KeyboardInterrupt:
            print("\n[Stopped] User interrupted.")
            break

    total_duration = total_bytes / (SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH)
    print(f"\n\nDone! Captured {total_bytes} bytes (~{total_duration:.2f}s of audio).")
    
    # Save to WAV
    try:
        with wave.open(OUTPUT_FILE, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(frames))
        print(f"Successfully saved to: {os.path.abspath(OUTPUT_FILE)}")
    except Exception as e:
        print(f"Error saving WAV file: {e}")

    sock.close()

if __name__ == "__main__":
    main()
