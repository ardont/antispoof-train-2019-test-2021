import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

class ASVspoof2019Dataset(Dataset):
    def __init__(self, protocol_file, audio_dir, transform=None):
        """
        protocol_file: путь к .txt файлу (train.trn.txt, dev.trl.txt)
        audio_dir: папка с аудио (train/, dev/, eval/)
        """
        self.audio_dir = audio_dir
        self.transform = transform
        self.file_list = []
        self.labels = []

        with open(protocol_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                file_id, label = parts
                # label: 'bonafide' -> 0, 'spoof' -> 1
                label = 0 if label == 'bonafide' else 1
                audio_path = os.path.join(audio_dir, file_id + '.flac')
                if os.path.exists(audio_path):
                    self.file_list.append(audio_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.file_list[idx])
        # Ресемплинг до 16 кГц, если нужно
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        label = self.labels[idx]
        if self.transform:
            waveform = self.transform(waveform)
        return waveform, label