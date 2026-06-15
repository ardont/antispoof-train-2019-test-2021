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

def apply_mu_law(y, mu=255):
    """
    Применяет сжатие и расширение u-law (mu-law) для симуляции 8-битного кодека G.711.
    """
    if len(y) == 0:
        return y
    y_abs = np.abs(y)
    # Сжатие
    y_compressed = np.sign(y) * np.log(1.0 + mu * y_abs) / np.log(1.0 + mu)
    # Квантование до 8 бит (256 уровней)
    y_quantized = np.round((y_compressed + 1.0) * 127.5) / 127.5 - 1.0
    # Расширение
    y_expanded = np.sign(y_quantized) * ((1.0 + mu) ** np.abs(y_quantized) - 1.0) / mu
    return y_expanded

def apply_a_law(y, A=87.6):
    """
    Применяет сжатие и расширение A-law для симуляции 8-битного кодека G.711.
    """
    if len(y) == 0:
        return y
    y_abs = np.abs(y)
    cond1 = y_abs < 1.0 / A
    cond2 = y_abs >= 1.0 / A
    
    # Сжатие
    y_compressed = np.zeros_like(y)
    y_compressed[cond1] = np.sign(y[cond1]) * (A * y_abs[cond1]) / (1.0 + np.log(A))
    y_compressed[cond2] = np.sign(y[cond2]) * (1.0 + np.log(A * y_abs[cond2])) / (1.0 + np.log(A))
    
    # Квантование до 8 бит
    y_quantized = np.round((y_compressed + 1.0) * 127.5) / 127.5 - 1.0
    
    # Расширение
    y_expanded = np.zeros_like(y)
    cond1_q = np.abs(y_quantized) < 1.0 / (1.0 + np.log(A))
    cond2_q = np.abs(y_quantized) >= 1.0 / (1.0 + np.log(A))
    
    y_expanded[cond1_q] = np.sign(y_quantized[cond1_q]) * (np.abs(y_quantized[cond1_q]) * (1.0 + np.log(A))) / A
    y_expanded[cond2_q] = np.sign(y_quantized[cond2_q]) * np.exp(np.abs(y_quantized[cond2_q]) * (1.0 + np.log(A)) - 1.0) / A
    return y_expanded

def apply_telephony_nb(y, sr=16000):
    """
    Симулирует узкополосный телефонный канал (Narrowband PSTN/GSM, G.711):
    Ресемплинг до 8 кГц -> полосовая фильтрация (300-3400 Гц) -> кодек -> ресемплинг до 16 кГц.
    """
    if len(y) == 0:
        return y
    # Ресемплинг вниз до 8 кГц
    y_8k = apply_resample_codec(y, sr, target_sr=8000)
    
    # Фильтрация 300 - 3400 Гц
    y_filt = apply_bandpass_filter(y_8k, sr=8000, low_cutoff=300, high_cutoff=3400)
    
    # Применение кодека A-law / u-law
    if np.random.rand() < 0.5:
        y_codec = apply_a_law(y_filt)
    else:
        y_codec = apply_mu_law(y_filt)
        
    # Ресемплинг вверх до исходной частоты
    y_up = apply_resample_codec(y_codec, 8000, target_sr=sr)
    return y_up

def apply_telephony_wb(y, sr=16000):
    """
    Симулирует широкополосный телефонный канал (Wideband VoIP, G.722 / AMR-WB):
    Полосовой фильтр (50-7000 Гц) -> кодек A-law/u-law.
    """
    if len(y) == 0:
        return y
    y_filt = apply_bandpass_filter(y, sr=sr, low_cutoff=50, high_cutoff=7000)
    if np.random.rand() < 0.5:
        y_codec = apply_a_law(y_filt)
    else:
        y_codec = apply_mu_law(y_filt)
    return y_codec

def augment_audio(y, sr=16000):
    """
    Применяет цепочку случайных искажений канала, реверберации и шума.
    """
    y_aug = y.copy()
    
    # 1. Симуляция канала / кодека (80% вероятность)
    if np.random.rand() < 0.8:
        channel_type = np.random.choice(['telephony_nb', 'telephony_wb', 'lowpass', 'resample'])
        if channel_type == 'telephony_nb':
            y_aug = apply_telephony_nb(y_aug, sr)
        elif channel_type == 'telephony_wb':
            y_aug = apply_telephony_wb(y_aug, sr)
        elif channel_type == 'lowpass':
            y_aug = apply_lowpass_filter(y_aug, sr, cutoff=np.random.randint(3500, 6000))
        elif channel_type == 'resample':
            y_aug = apply_resample_codec(y_aug, sr, target_sr=np.random.choice([8000, 11025, 12000]))
            
    # 2. Реверберация (RIR) (50% вероятность)
    if np.random.rand() < 0.5:
        y_aug = apply_reverb(y_aug, sr, rt60=np.random.uniform(0.1, 0.4), delay_ms=np.random.randint(10, 50))
        
    # 3. Аддитивный белый шум (50% вероятность)
    if np.random.rand() < 0.5:
        y_aug = add_noise(y_aug, snr_db=np.random.uniform(10, 30))
        
    return y_aug
