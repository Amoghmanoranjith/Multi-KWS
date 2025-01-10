import os
import json
from typing import List, Union, Optional, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

class SpeechLabelDataset(Dataset):
    """
    A simple dataset for loading audio files and their corresponding labels
    based on a manifest file.
    """

    def __init__(self, manifest_filepath: Union[str, List[str]], labels: List[str], sample_rate: int = 16000):
        """
        Args:
            manifest_filepath: Path to a JSON manifest file or a list of them.
            labels: List of label names.
            sample_rate: Target sample rate for audio files.
        """
        if isinstance(manifest_filepath, str):
            manifest_filepath = [manifest_filepath]

        self.items = []
        for file in manifest_filepath:
            with open(file, 'r') as f:
                self.items.extend(json.loads(line) for line in f)

        self.label2id = {label: idx for idx, label in enumerate(labels)}
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        # Load audio file
        audio_path = os.path.expanduser(item['audio_filepath'])
        waveform, sr = torchaudio.load(audio_path)

        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform)

        # Fetch label
        label = self.label2id[item['command']]

        # Optionally include duration (if needed)
        return waveform, torch.tensor(label)

def collate_fn(batch):
    """
    Collate function to pad audio signals to the same length.
    """
    waveforms, labels = zip(*batch)
    lengths = torch.tensor([w.size(1) for w in waveforms])
    padded_waveforms = torch.nn.utils.rnn.pad_sequence([w.t() for w in waveforms], batch_first=True).permute(0, 2, 1)
    return padded_waveforms, lengths, torch.tensor(labels)

def get_dataloader(
    manifest_filepath: Union[str, List[str]],
    labels: List[str],
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    sample_rate: int = 16000,
):
    """
    Create a DataLoader for the SpeechLabelDataset.
    """
    dataset = SpeechLabelDataset(manifest_filepath, labels, sample_rate)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
