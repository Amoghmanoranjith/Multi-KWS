import torch
from torch.utils.data import Dataset
import torchaudio
import json

class AudioLabelDataset(Dataset):
    """
    PyTorch Dataset for loading audio files and their corresponding labels.

    Args:
        manifest_filepath (str): Path to the manifest JSON file.
        labels (list): List of possible labels.
        featurizer (callable): A callable object to process raw audio into features (e.g., Mel Spectrogram).
        max_duration (float, optional): Maximum duration of audio to include in the dataset.
        min_duration (float, optional): Minimum duration of audio to include in the dataset.
        trim (bool, optional): Whether to trim silence from the audio files.
        load_audio (bool, optional): Whether to load audio or just return labels.
    """

    def __init__(
        self,
        manifest_filepath,
        labels,
        featurizer,
        max_duration=None,
        min_duration=None,
        trim=False,
        load_audio=True,
    ):
        # Load and filter the manifest
        with open(manifest_filepath, "r") as f:
            data = [json.loads(line.strip()) for line in f]
        
        self.samples = []
        for sample in data:
            duration = sample.get("duration", None)
            if (min_duration is not None and duration < min_duration) or \
               (max_duration is not None and duration > max_duration):
                continue
            self.samples.append(sample)

        self.featurizer = featurizer
        self.trim = trim
        self.load_audio = load_audio

        # Prepare label mappings
        self.labels = labels
        self.label2id = {label: idx for idx, label in enumerate(labels)}
        self.id2label = {idx: label for idx, label in enumerate(labels)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        audio_filepath = sample["audio_filepath"]
        label = sample["label"]
        offset = sample.get("offset", 0)
        duration = sample.get("duration", None)

        # Load and preprocess audio
        if self.load_audio:
            waveform, sample_rate = torchaudio.load(audio_filepath)
            
            # Apply offset and duration
            if offset > 0 or duration is not None:
                start_frame = int(offset * sample_rate)
                end_frame = int((offset + duration) * sample_rate) if duration else None
                waveform = waveform[:, start_frame:end_frame]

            # Trim silence if enabled
            if self.trim:
                waveform, _ = torchaudio.transforms.Vad(sample_rate=sample_rate)(waveform)

            # Convert waveform to features
            features = self.featurizer(waveform)
            feature_length = features.shape[-1]
        else:
            features = None
            feature_length = 0

        # Convert label to ID
        label_id = self.label2id[label]

        return features, feature_length, torch.tensor(label_id).long()

