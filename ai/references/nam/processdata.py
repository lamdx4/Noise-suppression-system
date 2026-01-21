import numpy as np
import tensorflow as tf


frame_length = 1022
n_fft = 1022
frame_step = 300
n_sample = 8000
sample_rate = 8000
n_frame  = (n_sample - frame_length)//frame_step +1
n_freq = n_fft//2 + 1


def split_real_imag(spectrogram):
    """ Tách phần thực và ảo của phổ """
    real_part = tf.math.real(spectrogram)
    imag_part = tf.math.imag(spectrogram)
    stacked = tf.stack([real_part, imag_part], axis=-1)  # (time, freq, 2)

    return stacked
def prepare_data(data, sample_rate=8000):
    """
    Prepare audio data for processing.

    Args:
        data (bytes): Raw audio data.
        sample_rate (int): Sample rate of the audio data.

    Returns:
        numpy.ndarray: Processed audio data as a numpy array.
    """

    spectrogram = tf.signal.stft(data, frame_length=frame_length, frame_step=frame_step, fft_length=n_fft)
    stacked = split_real_imag(spectrogram)
    stacked = tf.expand_dims(stacked, axis=0)

    # Resample if necessary (not implemented here, but can be done using scipy or librosa)

    return stacked
def istft(spectrogram, frame_length=frame_length, frame_step=frame_step, fft_length=n_fft):
    """
    Inverse Short-Time Fourier Transform (iSTFT) to convert spectrogram back to time-domain signal.

    Args:
        spectrogram (numpy.ndarray): Spectrogram data.
        frame_length (int): Length of each frame.
        frame_step (int): Step size between frames.
        fft_length (int): Length of FFT.

    Returns:
        numpy.ndarray: Time-domain signal.
    """
    # Convert stacked real and imaginary parts back to complex
    complex_spectrogram = tf.complex(spectrogram[..., 0], spectrogram[..., 1])

    # Perform iSTFT
    time_signal = tf.signal.inverse_stft(complex_spectrogram, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)

    return time_signal.numpy()
