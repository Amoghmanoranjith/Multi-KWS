# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import os
import random
from glob import glob
import shutil
import requests
import tarfile
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from transforms import MFCC, MelSpectrogram

label_dict = {
    "_silence_": 0,
    "_unknown_": 1,
    "down": 2,
    "go": 3,
    "left": 4,
    "no": 5,
    "off": 6,
    "on": 7,
    "right": 8,
    "stop": 9,
    "up": 10,
    "yes": 11,
}

DEFAULT_LABELS = [
    "_silence_",
    "_unknown_",
    "down",
    "go",
    "left",
    "no",
    "off",
    "on",
    "right",
    "stop",
    "up",
    "yes"
]
print("labels:\t", label_dict)
sample_per_cls_v1 = [1854, 258, 257]
sample_per_cls_v2 = [3077, 371, 408]
SR = 16000


def ScanAudioFiles(root_dir, ver):
    sample_per_cls = sample_per_cls_v1 if ver == 1 else sample_per_cls_v2
    audio_paths, labels = [], []
    for path, _, files in sorted(os.walk(root_dir, followlinks=True)):
        random.shuffle(files)
        for idx, filename in enumerate(files):
            if not filename.endswith(".wav"):
                continue
            dataset, class_name = path.split("/")[-2:]
            if class_name in ("_unknown_", "_silence_"):  # balancing
                if "train" in dataset and idx == sample_per_cls[0]:
                    break
                if "valid" in dataset and idx == sample_per_cls[1]:
                    break
                if "test" in dataset and idx == sample_per_cls[2]:
                    break
            audio_paths.append(os.path.join(path, filename))
            labels.append(label_dict[class_name])
    return audio_paths, labels


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
        targets.append(label)
    tensors = pad_sequence(tensors)
    targets = torch.LongTensor(targets)
    return tensors, targets

SAMPLE_RATE = 16000


class SpeechCommand(Dataset):
  """
  GSC
  subset : "training", "testing", or "validation"
  mode : "mfcc", "melspec", or "raw"
  ver : 1 or 2
  based on mode spec augmentation is applied
  before
  """
  def __init__(self, root_dir, noise_dir, subset, mode, augment, ver):
    self.data_list, self.labels = ScanAudioFiles(root_dir, ver)
    self.mode = mode
    self.subset = subset
    self.ver = ver
    self.augment = augment
    self.noise_dir = noise_dir
    self._noise = []
    
    # load all the noise waveforms for adding to the background
    for file_name in os.listdir(self.noise_dir):
          if file_name.endswith(".wav"):
              noise_path = os.path.join(self.noise_dir, file_name)
              noise_waveform, noise_sr = torchaudio.load(noise_path)
              noise_waveform = torchaudio.transforms.Resample(orig_freq=noise_sr, new_freq=SAMPLE_RATE)(noise_waveform)
              self._noise.append(noise_waveform)

    self.melkwargs={
    'n_fft': 480,                      # window_size
    'hop_length': 160,                # stride
    'n_mels': 80,                     # filterbank_channel_count
    'f_min': 20,                      # lower_frequency_limit
    'f_max': 7600,                    # upper_frequency_limit
    'power': 1.0,                     # magnitude_squared = False
    'mel_scale': 'htk'}
    self.augkwargs = {
    'freq_mask': 7,
    'time_mask': 25,
    'num_time_mask': 2,
    'num_freq_mask': 2
    } if augment else None
    self.min_ratio = 0.85
    self.max_ratio = 1.15
    if mode == "mfcc":
        self.to_mel = MFCC(
              sample_rate=16000,
              n_mfcc=40,
              log_mels = True,
              augkwargs = self.augkwargs,
              melkwargs=self.melkwargs)
    if mode == "melspec":
        self.to_mel = MelSpectrogram(
              sample_rate=16000,
              **self.melkwargs)

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    audio_path = self.data_list[idx]
    sample, sample_rate = torchaudio.load(audio_path)
    label = self.labels[idx]
    # apply resampling if augmentation allowed
    if self.augment:
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        new_sr = int(SAMPLE_RATE * ratio)
        sample = transforms.Resample(orig_freq=sample_rate, new_freq=new_sr)
    if sample_rate != SAMPLE_RATE:
      resampler = transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
      waveform = resampler(waveform)
    if self.augment:
        # add background sound
      sample = self._noise_augment(sample, foreground_volume = 1.0, background_volume = 0.1)
      sample = self._shift_augment(sample)
    if self.mode == "raw":
      return sample, label
    # convert to MFCC
    sample = self.to_mel(sample)
    return sample, label

  def _noise_augment(self, waveform, foreground_volume, background_volume):
    noise_waveform = random.choice(self._noise)
    noise_sample_start = 0
    if noise_waveform.shape[1] - waveform.shape[1] > 0:
        noise_sample_start = random.randint(0, noise_waveform.shape[1] - waveform.shape[1])
    noise_waveform = noise_waveform[:, noise_sample_start:noise_sample_start+waveform.shape[1]]
    clamp = noise_waveform * background_volume
    clamp += waveform * foreground_volume
    return clamp

  def _shift_augment(self, waveform):
    shift = random.randint(-100, 100)
    waveform = torch.roll(waveform, shift)
    if shift > 0:
        waveform[0][:shift] = 0
    elif shift < 0:
        waveform[0][shift:] = 0
    return waveform


def DownloadDataset(loc, url):
    if not os.path.isdir(loc):
        os.mkdir(loc)
    filename = os.path.basename(url)
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1048576  # 1MB

    with open(os.path.join(loc, filename), "wb") as f, \
         tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
        for data in response.iter_content(block_size):
            f.write(data)
            pbar.update(len(data))

    with tarfile.open(os.path.join(loc, filename), "r:gz") as tar:
        print("Extracting dataset...")
        tar.extractall(loc)

def make_empty_audio(loc, num):
    if not os.path.isdir(loc):
        os.mkdir(loc)
    for i in range(num):
        path = os.path.join(loc, "%s.wav" % str(i))
        zeros = torch.zeros([1, SR])  # 1 sec long.
        torchaudio.save(path, zeros, SR)


def make_12class_dataset(base, target):
    os.mkdir(target)
    os.mkdir(target + "/_unknown_")
    class10 = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]
    for clsdir in tqdm(glob(os.path.join(base, "*")), desc="Processing classes"):
        class_name = os.path.basename(clsdir)
        if class_name in class10:
            target_dir = os.path.join(target, class_name)
            shutil.copytree(clsdir, target_dir)
        else:
            for file_path in glob(os.path.join(clsdir, "*")):
                filename = os.path.basename(file_path)
                target_dir = os.path.join(target, "_unknown_")
                os.makedirs(target_dir, exist_ok=True)
                target_file = os.path.join(target_dir, class_name + "_" + filename)
                shutil.copy(file_path, target_file)

def split_data(base, target, valid_list, test_list):
    with open(valid_list, "r") as f:
        valid_names = [item.rstrip() for item in f.readlines()]
    with open(test_list, "r") as f:
        test_names = [item.rstrip() for item in f.readlines()]

    trg_base_dirs = [
        os.path.join(target, "train"),
        os.path.join(target, "valid"),
        os.path.join(target, "test"),
    ]
    for item in trg_base_dirs:
        if not os.path.isdir(item):
            os.mkdir(item)

    all_files = []
    for root, _, files in os.walk(base):
        for file_name in files:
            if file_name.endswith(".wav") and "_background_noise_" not in os.path.join(root, file_name):
                all_files.append((root, file_name))

    for root, file_name in tqdm(all_files, desc="Splitting dataset"):
        class_name = os.path.basename(root)
        for item in trg_base_dirs:
            class_path = os.path.join(item, class_name)
            if not os.path.isdir(class_path):
                os.mkdir(class_path)

        org_file_name = os.path.join(root, file_name)
        trg_file_name = os.path.join(class_name, file_name)
        if trg_file_name in valid_names:
            target_dir = trg_base_dirs[1]
        elif trg_file_name in test_names:
            target_dir = trg_base_dirs[-1]
        else:
            target_dir = trg_base_dirs[0]

        target_path = os.path.join(target_dir, trg_file_name)
        shutil.copy(org_file_name, target_path)


def SplitDataset(loc):
    target_loc = "%s_split" % loc
    if not os.path.isdir(target_loc):
        os.mkdir(target_loc)
    split_data(
        loc,
        target_loc,
        os.path.join(loc, "validation_list.txt"),
        os.path.join(loc, "testing_list.txt"),
    )

    sample_per_cls = sample_per_cls_v1 if "v0.01" in loc else sample_per_cls_v2
    for idx, split_name in enumerate(["train", "valid", "test"]):
        make_12class_dataset(
            "%s/%s" % (target_loc, split_name), "%s/%s_12class" % (loc, split_name)
        )
        make_empty_audio("%s/%s_12class/_silence_" % (loc, split_name), sample_per_cls[idx])

def _load_data(ver, download = True):
    """
    method that loads data into the object.
    Downloads and splits the data if necessary.
    """
    print("Check google speech commands dataset v1 or v2 ...")
    if not os.path.isdir("./data"):
        os.mkdir("./data")
    base_dir = "./data/speech_commands_v0.01"
    url = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz"
    url_test = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_test_set_v0.01.tar.gz"
    if ver == 2:
        base_dir = base_dir.replace("v0.01", "v0.02")
        url = url.replace("v0.01", "v0.02")
        url_test = url_test.replace("v0.01", "v0.02")
    test_dir = base_dir.replace("commands", "commands_test_set")
    if download:
        old_dirs = glob(base_dir.replace("commands_", "commands_*"))
        for old_dir in old_dirs:
            shutil.rmtree(old_dir)
        os.mkdir(test_dir)
        DownloadDataset(test_dir, url_test)
        os.mkdir(base_dir)
        DownloadDataset(base_dir, url)
        SplitDataset(base_dir)
        print("Done...")

    # Define data directories
    train_dir = "%s/train_12class" % base_dir
    valid_dir = "%s/valid_12class" % base_dir
    noise_dir = "%s/_background_noise_" % base_dir

    return train_dir, test_dir, valid_dir, noise_dir
