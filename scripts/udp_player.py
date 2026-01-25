import socket
import pyaudio
import collections
import struct

# Configuration
UDP_IP = "0.0.0.0"
UDP_PORT = 12345
SAMPLE_RATE = 48000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 480  # Buffer size (10ms at 48kHz)

def main():
    # Initialize UDP Socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    
    # Increase socket buffer to avoid drops
    snd_buf_size = 32 * 1024
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, snd_buf_size)
    
    print(f"Listening for UDP audio on {UDP_IP}:{UDP_PORT}...")

    # Initialize PyAudio
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    output=True,
                    frames_per_buffer=CHUNK)

    print("Playing audio...")

    # Jitter Buffer Config
    BUFFER_SIZE = 2 # Reduced buffer for faster start (20ms latency)
    jitter_buffer = collections.deque()
    chk_underflow = False
    
    try:
        while True:
            # Receive packet
            data, addr = sock.recvfrom(4096)
            
            # Add to buffer
            jitter_buffer.append(data)
            
            # Only play when buffer is full enough (Prefill)
            if len(jitter_buffer) >= BUFFER_SIZE:
                if not chk_underflow:
                    print("Buffering complete. Playing!")
                stream.write(jitter_buffer.popleft(), exception_on_underflow=False)
                chk_underflow = True
            elif chk_underflow:
                 # If we ran dry after starting, we are underflowing. 
                 # Optional: Insert silence or just wait (we just wait here)
                 pass

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        sock.close()

if __name__ == "__main__":
    main()
