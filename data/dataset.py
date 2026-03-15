"""
Dataset classes for BirdCLEF+ 2026.

Handles loading soundscape audio, slicing into 5-second chunks,
and converting to mel spectrograms for model training.
"""

import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


# ─── Audio constants (match Kaggle's expected format) ───
SAMPLE_RATE = 32000
CHUNK_DURATION = 5  # seconds
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_DURATION  # 160,000

# ─── Mel spectrogram defaults ───
N_FFT = 2048
HOP_LENGTH = 512  # ~16ms at 32kHz
N_MELS = 128
F_MIN = 60
F_MAX = 16000


def load_audio(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio file, resample to target sr, convert to mono.

    Uses torchaudio for speed. Falls back to librosa if needed.
    Returns numpy array of shape (n_samples,).
    """
    import torchaudio

    waveform, orig_sr = torchaudio.load(str(path))

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if orig_sr != sr:
        resampler = torchaudio.transforms.Resample(orig_sr, sr)
        waveform = resampler(waveform)

    return waveform.squeeze(0).numpy()


def audio_to_melspec(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
    f_min: float = F_MIN,
    f_max: float = F_MAX,
) -> np.ndarray:
    """Convert audio waveform to log mel spectrogram.

    Returns numpy array of shape (n_mels, time_frames).
    """
    import librosa

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max,
        power=2.0,
    )
    # Log scale with small offset to avoid log(0)
    log_mel = np.log(mel + 1e-9)
    return log_mel.astype(np.float32)


class BirdCLEFDataset(Dataset):
    """Dataset for BirdCLEF+ 2026 training.

    Each sample is a 5-second audio chunk converted to a mel spectrogram,
    with a multi-label target vector over all species.

    Args:
        audio_dir: Path to directory containing .ogg audio files.
        metadata_csv: Path to CSV with columns [filename, species, ...].
        species_list: Ordered list of all species names.
        transform: Optional callable for data augmentation.
    """

    def __init__(
        self,
        audio_dir: str | Path,
        metadata_csv: str | Path,
        species_list: list[str],
        transform=None,
    ):
        import pandas as pd

        self.audio_dir = Path(audio_dir)
        self.species_list = species_list
        self.species_to_idx = {s: i for i, s in enumerate(species_list)}
        self.transform = transform

        self.metadata = pd.read_csv(metadata_csv)
        self._build_index()

    def _build_index(self):
        """Build sample index: list of (audio_path, label_indices)."""
        self.samples = []
        for _, row in self.metadata.iterrows():
            path = self.audio_dir / row["filename"]
            if not path.exists():
                continue
            # Handle single or multi-label
            species = row.get("primary_label", row.get("species", ""))
            if isinstance(species, str):
                species_names = [s.strip() for s in species.split(";")]
            else:
                species_names = [str(species)]
            label_idxs = [self.species_to_idx[s] for s in species_names if s in self.species_to_idx]
            if label_idxs:
                self.samples.append((path, label_idxs))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_idxs = self.samples[idx]

        # Load and process audio
        audio = load_audio(path)

        # Pad or crop to exactly CHUNK_SAMPLES
        if len(audio) < CHUNK_SAMPLES:
            audio = np.pad(audio, (0, CHUNK_SAMPLES - len(audio)))
        elif len(audio) > CHUNK_SAMPLES:
            # Random crop during training
            start = np.random.randint(0, len(audio) - CHUNK_SAMPLES)
            audio = audio[start : start + CHUNK_SAMPLES]

        # Convert to mel spectrogram
        mel = audio_to_melspec(audio)

        # Build multi-label target
        target = np.zeros(len(self.species_list), dtype=np.float32)
        for idx_label in label_idxs:
            target[idx_label] = 1.0

        mel_tensor = torch.from_numpy(mel).unsqueeze(0)  # (1, n_mels, time)
        target_tensor = torch.from_numpy(target)

        if self.transform:
            mel_tensor = self.transform(mel_tensor)

        return mel_tensor, target_tensor


class SoundscapeDataset(Dataset):
    """Dataset for inference on soundscape recordings.

    Slices soundscapes into non-overlapping 5-second chunks and generates
    row_ids matching the Kaggle submission format: "{filename}_{end_time}".

    Args:
        soundscape_dir: Path to directory containing soundscape .ogg files.
    """

    def __init__(self, soundscape_dir: str | Path):
        self.soundscape_dir = Path(soundscape_dir)
        self._build_index()

    def _build_index(self):
        """Build index of all 5-second chunks across all soundscapes."""
        self.chunks = []  # list of (filepath, start_sample, end_time_seconds, row_id)

        audio_files = sorted(self.soundscape_dir.glob("*.ogg"))
        for audio_path in audio_files:
            stem = audio_path.stem
            # Estimate duration without loading full file
            import torchaudio

            info = torchaudio.info(str(audio_path))
            duration_sec = info.num_frames / info.sample_rate
            n_chunks = math.ceil(duration_sec / CHUNK_DURATION)

            for i in range(n_chunks):
                start_sample = i * CHUNK_SAMPLES
                end_time = (i + 1) * CHUNK_DURATION
                row_id = f"{stem}_{end_time}"
                self.chunks.append((audio_path, start_sample, end_time, row_id))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        audio_path, start_sample, end_time, row_id = self.chunks[idx]

        audio = load_audio(audio_path)

        # Extract chunk
        chunk = audio[start_sample : start_sample + CHUNK_SAMPLES]

        # Pad if at the end of the file
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))

        mel = audio_to_melspec(chunk)
        mel_tensor = torch.from_numpy(mel).unsqueeze(0)  # (1, n_mels, time)

        return mel_tensor, row_id
