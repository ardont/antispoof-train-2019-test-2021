import numpy as np
from scipy import signal
import librosa

def apply_bandpass_filter(y, sr=16000, low_cutoff=300, high_cutoff=3400):
    """
    Simulates telephone bandpass (GSM codec, etc.).
    """
    if len(y) == 0:
        return y
    nyquist = 0.5 * sr
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    return signal.filtfilt(b, a, y)

def apply_lowpass_filter(y, sr=16000, cutoff=4000):
    """
    Simulates high-frequency attenuation from compression/codecs.
    """
    if len(y) == 0:
        return y
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(4, normal_cutoff, btype='low')
    return signal.filtfilt(b, a, y)

def apply_resample_codec(y, sr=16000, target_sr=8000):
    """
    Simulates downsampling/upsampling codec compression using fast polyphase resampling.
    """
    if len(y) == 0:
        return y
    import math
    gcd = math.gcd(sr, target_sr)
    p = target_sr // gcd
    q = sr // gcd
    
    # Resample down and back up using scipy's fast polyphase resampler
    y_down = signal.resample_poly(y, p, q)
    y_up = signal.resample_poly(y_down, q, p)
    
    # Ensure length matches original
    if len(y_up) < len(y):
        y_up = np.pad(y_up, (0, len(y) - len(y_up)))
    else:
        y_up = y_up[:len(y)]
    return y_up

def apply_reverb(y, sr=16000, rt60=0.2, delay_ms=30):
    """
    Simulates Room Impulse Response (RIR) reverberation.
    """
    if len(y) == 0:
        return y
    # Generate decay tail
    t = np.arange(0, rt60, 1.0 / sr)
    decay = np.exp(-6 * np.log(10) * t / rt60)
    
    # Noise shaped by decay envelope
    rir = np.random.randn(len(t)) * decay
    
    # Add direct path at start
    delay_samples = int(sr * delay_ms / 1000)
    rir = np.concatenate([np.zeros(delay_samples), rir])
    rir[0] = 1.0
    
    # Normalize energy
    rir = rir / np.linalg.norm(rir)
    
    # Convolve using fast FFT convolution
    y_rev = signal.fftconvolve(y, rir, mode='same')
    return y_rev

def add_noise(y, snr_db=20):
    """
    Adds white noise to the audio.
    """
    if len(y) == 0:
        return y
    sig_power = np.mean(y ** 2)
    if sig_power <= 0:
        return y
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(y))
    return y + noise

def augment_audio(y, sr=16000):
    """
    Randomly applies one or more of the augmentations to the input audio.
    """
    y_aug = y.copy()
    
    # 1. Channel / Codec Simulation (60% chance)
    if np.random.rand() < 0.6:
        channel_type = np.random.choice(['bandpass', 'lowpass', 'resample'])
        if channel_type == 'bandpass':
            # GSM-like
            y_aug = apply_bandpass_filter(y_aug, sr, low_cutoff=np.random.randint(250, 350), high_cutoff=np.random.randint(3000, 3600))
        elif channel_type == 'lowpass':
            # Lossy codec low-pass
            y_aug = apply_lowpass_filter(y_aug, sr, cutoff=np.random.randint(3500, 6000))
        elif channel_type == 'resample':
            # Bitrate downsampling
            y_aug = apply_resample_codec(y_aug, sr, target_sr=np.random.choice([8000, 11025, 12000]))
            
    # 2. Reverberation (RIR) (40% chance)
    if np.random.rand() < 0.4:
        y_aug = apply_reverb(y_aug, sr, rt60=np.random.uniform(0.1, 0.35), delay_ms=np.random.randint(15, 45))
        
    # 3. Add Noise (40% chance)
    if np.random.rand() < 0.4:
        y_aug = add_noise(y_aug, snr_db=np.random.uniform(15, 30))
        
    return y_aug
