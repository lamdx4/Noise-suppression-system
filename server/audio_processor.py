import ctypes
import numpy as np

FRAME_SIZE = 480

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
