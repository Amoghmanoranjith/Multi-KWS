import os
import json
from typing import List, Union, Optional, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

class SpeechLabelDataset(Dataset):
    """
    A simple dataset for loading audio files and their corresponding labels
    based on a manifest file, with audio samples truncated/padded to 1 second.
    """

    def __init__(self, manifest_filepath: Union[str, List[str]], labels: List[str], sample_rate: int = 16000, target_duration: float = 1.0):
        """
        Args:
            manifest_filepath: Path to a JSON manifest file or a list of them.
            labels: List of label names.
            sample_rate: Target sample rate for audio files.
            target_duration: Duration in seconds to which all audio samples will be padded/truncated.
        """
        if isinstance(manifest_filepath, str):
            manifest_filepath = [manifest_filepath]

        self.items = []
        for file in manifest_filepath:
            with open(file, 'r') as f:
                self.items.extend(json.loads(line) for line in f)

        self.label2id = {label: idx for idx, label in enumerate(labels)}
        self.sample_rate = sample_rate
        self.target_length = int(target_duration * sample_rate)  # Fixed length for 1 second of audio

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

        # Truncate or pad to the target length
        if waveform.size(1) > self.target_length:
            waveform = waveform[:, :self.target_length]  # Truncate
        elif waveform.size(1) < self.target_length:
            padding = self.target_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, padding))  # Pad

        # Fetch label
        label = self.label2id[item['command']]
        return waveform, torch.tensor(label)

def seq_collate_fn(batch, token_pad_value=0):
    """Collate batch of audio signals, audio lengths, tokens, and token lengths.
    
    This assumes the signals are already padded/truncated to a fixed length (1 second).
    
    Args:
        batch: A list of tuples where each tuple contains:
            - audio signal (Tensor): the waveform
            - audio length (Tensor): the length of the waveform
            - tokens (Tensor): the tokenized audio
            - tokens length (Tensor): the length of the token sequence
    Returns:
        A tuple of (audio_signal, audio_lengths, tokens, tokens_lengths):
            - audio_signal: Batch of audio signals (Tensor)
            - audio_lengths: Lengths of each audio signal (Tensor)
            - tokens: Batch of tokenized sequences (Tensor)
            - tokens_lengths: Lengths of each token sequence (Tensor)
    """
    
    _, audio_lengths, _, tokens_lengths = zip(*batch)

    # Since all audio signals should be 1 second (16000 samples), we can set max_audio_len to 16000
    max_audio_len = 16000  
    max_tokens_len = max(tokens_lengths).item()

    audio_signal, tokens = [], []
    
    for sig, sig_len, tokens_i, tokens_i_len in batch:
        # Audio signals are expected to be 1 second, so no further padding or truncation for them
        audio_signal.append(sig)
        
        # Pad tokens to the max length
        tokens_i_len = tokens_i_len.item()
        if tokens_i_len < max_tokens_len:
            pad = (0, max_tokens_len - tokens_i_len)
            tokens_i = torch.nn.functional.pad(tokens_i, pad, value=token_pad_value)
        tokens.append(tokens_i)

    # Stack audio signals into a tensor (batch)
    audio_signal = torch.stack(audio_signal)
    audio_lengths = torch.stack(audio_lengths)

    # Stack token sequences into a tensor (batch)
    tokens = torch.stack(tokens)
    tokens_lengths = torch.stack(tokens_lengths)

    return audio_signal, audio_lengths, tokens, tokens_lengths

def get_dataloader(
    manifest_filepath: Union[str, List[str]],
    labels: List[str],
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    sample_rate: int = 16000,
    target_duration: float = 1.0,  # 1 second duration for each sample
):
    dataset = SpeechLabelDataset(manifest_filepath, labels, sample_rate, target_duration)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=seq_collate_fn,  # Use the seq_collate_fn here
    )
