import numpy as np
import scipy.fftpack
import librosa

_filterbank_cache = {}

def get_linear_filterbank(sr, n_fft, n_filters=128):
    """
    Creates a linear scale filterbank (cached for performance).
    """
    key = (sr, n_fft, n_filters)
    if key in _filterbank_cache:
        return _filterbank_cache[key]
        
    # Frequencies of the FFT bins
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    
    # Linear spacing of filter center frequencies from 0 to Nyquist (sr / 2)
    filter_freqs = np.linspace(0, sr / 2, n_filters + 2)
    
    weights = np.zeros((n_filters, n_fft // 2 + 1))
    for i in range(n_filters):
        lower = filter_freqs[i]
        center = filter_freqs[i+1]
        upper = filter_freqs[i+2]
        
        # Ascending ramp
        if center > lower:
            ascending = (freqs - lower) / (center - lower)
        else:
            ascending = np.zeros_like(freqs)
            
        # Descending ramp
        if upper > center:
            descending = (upper - freqs) / (upper - center)
        else:
            descending = np.zeros_like(freqs)
            
        weights[i] = np.maximum(0, np.minimum(ascending, descending))
        
    _filterbank_cache[key] = weights
    return weights

def extract_lfcc(y, sr=16000, n_lfcc=20, n_filters=128, n_fft=400, hop_length=160, win_length=400):
    """
    Extracts custom Linear Frequency Cepstral Coefficients (LFCC).
    y: 1D audio signal (numpy array)
    """
    # STFT to get the magnitude spectrogram
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
    power_spec = stft ** 2
    
    # Build filterbank
    fb = get_linear_filterbank(sr, n_fft, n_filters)
    
    # Apply filterbank
    linear_spec = np.dot(fb, power_spec)
    
    # Log scale
    log_linear_spec = np.log(linear_spec + 1e-10)
    
    # Discrete Cosine Transform (type 2, normalized)
    lfcc = scipy.fftpack.dct(log_linear_spec, axis=0, type=2, norm='ortho')
    
    # Keep the first n_lfcc coefficients
    return lfcc[:n_lfcc, :]
