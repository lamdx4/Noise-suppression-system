import ctypes
import numpy as np
import os

DLL_PATH = "./rnnoise.dll"
FRAME_SIZE = 480

def test_binding():
    print(f"Loading {DLL_PATH}...")
    if not os.path.exists(DLL_PATH):
        print("Error: DLL not found!")
        return

    try:
        lib = ctypes.CDLL(DLL_PATH)
        print("DLL loaded successfully.")
        
        lib.rnnoise_create.restype = ctypes.c_void_p
        lib.rnnoise_create.argtypes = [ctypes.c_void_p]
        
        lib.rnnoise_process_frame.restype = ctypes.c_float
        lib.rnnoise_process_frame.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        
        lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
        
        print("Creating RNNoise state...")
        st = lib.rnnoise_create(None)
        if not st:
            print("Error: Failed to create state.")
            return
            
        print("Processing test frame...")
        data = np.zeros(FRAME_SIZE, dtype=np.float32)
        in_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        out_ptr = (ctypes.c_float * FRAME_SIZE)()
        
        vad_prob = lib.rnnoise_process_frame(st, out_ptr, in_ptr)
        print(f"VAD Prob for silent frame: {vad_prob}")
        
        print("Destroying state...")
        lib.rnnoise_destroy(st)
        print("Test completed successfully.")
        
    except Exception as e:
        print(f"Error during binding test: {e}")

if __name__ == "__main__":
    test_binding()
