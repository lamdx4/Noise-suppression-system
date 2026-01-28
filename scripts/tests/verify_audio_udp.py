
import socket
import wave
import time
import os

# Configuration
UDP_IP = "0.0.0.0"  # Listen on all interfaces
UDP_PORT = 12345    # Must match ESP32 target port
DURATION = 10       # Recording duration in seconds
OUTPUT_FILE = "recorded_rnnoise_output.wav"

# Audio Format (Standard RNNoise / ESP32 I2S settings)
SAMPLE_RATE = 48000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit = 2 bytes

def main():
    print(f"--- UDP Audio Recorder Tool ---")
    print(f"Listening on {UDP_IP}:{UDP_PORT}")
    print(f"Format: {SAMPLE_RATE}Hz, {CHANNELS} Channel, 16-bit PCM")
    print(f"Target Duration: {DURATION} seconds")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((UDP_IP, UDP_PORT))
    except Exception as e:
        print(f"Error binding to port: {e}")
        return

    frames = []
    
    print("\n[Waiting for data...]")
    
    # Wait for first packet to start timer
    data, addr = sock.recvfrom(4096)
    frames.append(data)
    print(f"[Started] Receiving from {addr}")
    
    start_time = time.time()
    packet_count = 1
    total_bytes = len(data)

    while True:
        elapsed = time.time() - start_time
        if elapsed >= DURATION:
            break
            
        try:
            sock.settimeout(1.0) # Timeout if stream stops
            data, _ = sock.recvfrom(4096)
            frames.append(data)
            packet_count += 1
            total_bytes += len(data)
            
            # Progress indicator
            if packet_count % 50 == 0:
                print(f"\rRecording... {elapsed:.1f}s / {DURATION}s ({packet_count} packets)", end="")
        except socket.timeout:
            print("\n[Timeout] Stream stopped sending data.")
            break
        except KeyboardInterrupt:
            print("\n[Stopped] User interrupted.")
            break

    print(f"\n\nDone! Captured {packet_count} packets ({total_bytes} bytes).")
    
    # Save to WAV
    try:
        with wave.open(OUTPUT_FILE, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(frames))
        print(f"Successfully saved to: {os.path.abspath(OUTPUT_FILE)}")
        print("\nAnalyze this file to check:")
        print("1. If audio sounds clear but slow/choppy -> Real-time Latency issue (Underrun).")
        print("2. If audio sounds strictly noise/static -> Decoding/Endianness issue.")
    except Exception as e:
        print(f"Error saving WAV file: {e}")

    sock.close()

if __name__ == "__main__":
    main()
