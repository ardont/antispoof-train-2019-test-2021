import numpy as np
import scipy.fftpack
import librosa

def extract_cqcc(y, sr=16000, n_cqcc=20, hop_length=160, fmin=300, n_bins=42, bins_per_octave=12):
    """
    Extracts robust Constant Q Cepstral Coefficients (CQCC) restricted to [fmin, fmax] frequency band.
    y: 1D audio signal (numpy array)
    """
    # Compute the Constant Q Transform (magnitude only)
    cqt = np.abs(librosa.cqt(
        y, sr=sr, hop_length=hop_length, fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave
    ))
    
    # Power spectrogram
    power_cqt = cqt ** 2
    
    # Log scale
    log_power_cqt = np.log(power_cqt + 1e-10)
    
    # Discrete Cosine Transform (type 2, normalized)
    cqcc = scipy.fftpack.dct(log_power_cqt, axis=0, type=2, norm='ortho')
    
    # Keep the first n_cqcc coefficients
    return cqcc[:n_cqcc, :]
