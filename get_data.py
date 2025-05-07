import os
import math
import random
import torch
import torchaudio
from torchaudio import transforms
from torchaudio.datasets import SPEECHCOMMANDS


EPS = 1e-9
SAMPLE_RATE = 16000


# default labels from GSC dataset
# define this as per requirement
DEFAULT_LABELS = ['visual', 'wow', 'learn', 'backward', 'dog', 'two', 'left', 'happy', 'nine', 'go', 'up', 'bed', 'stop', 'one',
     'zero', 'tree', 'seven', 'on', 'four', 'bird', 'right', 'eight', 'no', 'six', 'forward', 'house', 'marvin',
     'sheila', 'five', 'off', 'three', 'down', 'cat', 'follow', 'yes', 'silence']

N_CLASS = len(DEFAULT_LABELS)

def get_size(model):
     keys = model.state_dict().keys()
     total_size = 0
     for i in keys:
         if model.state_dict()[i].shape:
             # print(torch.tensor(model.state_dict()[i].shape))
             total_size += model.state_dict()[i].shape.numel()
     total_size *= 4
     total_size /= 1e6
     print(f"model size: {total_size} MB")

def prepare_wav(waveform, sample_rate):
    if sample_rate != SAMPLE_RATE: 
        resampler = transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)
    to_mel = transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, f_max=8000, n_mels=40)
    log_mel = (to_mel(waveform) + EPS).log2()
    return log_mel


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str, path="./", mode = 'mel', MelParams = None):
        super().__init__(path, download=True)
        self.mode = mode
         
        self.melkwargs={
        'n_fft': 480,                      # window_size
        'hop_length': 160,                # stride
        'n_mels': 80,                     # filterbank_channel_count
        'f_min': 20,                      # lower_frequency_limit
        'f_max': 7600,                    # upper_frequency_limit
        'power': 1.0,                     # magnitude_squared = False
        'mel_scale': 'htk'}
        self.to_mel = transforms.MFCC(
                  sample_rate=16000,
                  n_mfcc=40,
                  log_mels = True,
                  melkwargs=melkwargs)
        if MelParams:
             self.to_mel = transforms.MelSpectrogram(**MelParams)
        self.subset = subset

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fh:
                return [
                    os.path.join(self._path, line.strip()) for line in fh
                ]

        self._noise = []

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

            noise_paths = [w for w in os.listdir(os.path.join(self._path, "_background_noise_")) if w.endswith(".wav")]
            for item in noise_paths:
                noise_path =  os.path.join(self._path, "_background_noise_", item)
                noise_waveform, noise_sr = torchaudio.load(noise_path)
                noise_waveform = transforms.Resample(orig_freq=noise_sr, new_freq=SAMPLE_RATE)(noise_waveform)
                self._noise.append(noise_waveform)
        else:
            raise ValueError(f"Unknown subset {subset}. Use validation/testing/training")

    def _noise_augment(self, waveform):
        noise_waveform = random.choice(self._noise)

        noise_sample_start = 0
        if noise_waveform.shape[1] - waveform.shape[1] > 0:
            noise_sample_start = random.randint(0, noise_waveform.shape[1] - waveform.shape[1])
        noise_waveform = noise_waveform[:, noise_sample_start:noise_sample_start+waveform.shape[1]]

        signal_power = waveform.norm(p=2)
        noise_power = noise_waveform.norm(p=2)

        snr_dbs = [20, 10, 3]
        snr = random.choice(snr_dbs)

        snr = math.exp(snr / 10)
        scale = snr * noise_power / signal_power
        noise_waveform  = 0.1 * noise_waveform
        noisy_signal = (scale * waveform + noise_waveform) / 2
        return noisy_signal

    def _shift_augment(self, waveform):
         # change this according to time shift metric
        shift = random.randint(-100, 100)
        waveform = torch.roll(waveform, shift)
        if shift > 0:
            waveform[0][:shift] = 0
        elif shift < 0:
            waveform[0][shift:] = 0
        return waveform

    def _augment(self, waveform):
        if random.random() < 0.8:
            waveform = self._noise_augment(waveform)
        
        waveform = self._shift_augment(waveform)
        return waveform

    def __getitem__(self, n):
        waveform, sample_rate, label, _, _ = super().__getitem__(n)
        if sample_rate != SAMPLE_RATE: 
            resampler = transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)
        if self.mode == 'raw':
             return waveform, label
        if self.subset == "training":
            waveform = self._augment(waveform)
        log_mel = (self.to_mel(waveform) + EPS).log2()

        return log_mel, label


_label_to_idx = {label: i for i, label in enumerate(DEFAULT_LABELS)}
_idx_to_label = {i: label for label, i in _label_to_idx.items()}


def label_to_idx(label):
    return _label_to_idx[label]


def idx_to_label(idx):
    return _idx_to_label[idx]


def pad_sequence(batch):
    batch = [item.permute(2, 1, 0) for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
    return batch.permute(0, 3, 2, 1) 


def collate_fn(batch):
    tensors, targets = [], []
    for log_mel, label in batch:
        tensors.append(log_mel)
        targets.append(label_to_idx(label))

    tensors = pad_sequence(tensors)
    targets = torch.LongTensor(targets)

    return tensors, targets
