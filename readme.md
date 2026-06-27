# Audio Feature Extraction using Librosa

This project demonstrates how to load audio files and extract common
speech/audio features using **Librosa**.

## Features Covered

-   Load audio file (.flac / .wav)
-   Compute duration and sampling rate
-   Trim silence from audio
-   Compute Short-Time Fourier Transform (STFT)
-   Generate Mel Spectrogram
-   Extract MFCC features
-   Convert amplitude spectrogram to decibel (dB) scale
-   Visualize all features using Matplotlib

------------------------------------------------------------------------

## 📦 Requirements

``` bash
pip install librosa matplotlib numpy
```

------------------------------------------------------------------------

## 🚀 Workflow

### 1. Load Audio

``` python
audio, sr = librosa.load(audio_path, sr=None)
```

### 2. Duration

``` python
duration = librosa.get_duration(y=audio, sr=sr)
```

### 3. Trim Silence

``` python
speech_audio, _ = librosa.effects.trim(audio)
```

### 4. STFT (Frequency Representation)

``` python
D = librosa.stft(audio)
DB = librosa.amplitude_to_db(abs(D), ref=np.max)
```

### 5. Mel Spectrogram

``` python
mel = librosa.feature.melspectrogram(y=audio, sr=sr)
```

### 6. MFCC Features

``` python
mfcc = librosa.feature.mfcc(y=audio, sr=sr)
```

------------------------------------------------------------------------

## 📊 Visualization Dashboard

You can visualize: - Waveform (original & trimmed) - STFT Spectrogram -
Mel Spectrogram - MFCC Heatmap

Using:

``` python
import matplotlib.pyplot as plt
import librosa.display
```

------------------------------------------------------------------------

## 🔢 Key Concept: dB Conversion

``` python
DB = librosa.amplitude_to_db(np.abs(D), ref=np.max)
```

-   Converts amplitude → decibels (log scale)
-   `np.abs(D)` extracts magnitude from complex STFT
-   `ref=np.max` normalizes values so max = 0 dB

------------------------------------------------------------------------

## 🧠 Use Cases

-   Speech recognition
-   Speaker identification
-   Audio classification
-   Music analysis

------------------------------------------------------------------------

## 📌 Output Features

-   Sampling rate
-   Audio duration
-   Number of samples
-   Spectrograms
-   MFCC matrix

------------------------------------------------------------------------

## 📁 Example Pipeline

1.  Load audio\
2.  Trim silence\
3.  Extract STFT / Mel / MFCC\
4.  Convert to dB scale\
5.  Visualize results

------------------------------------------------------------------------

## 🧑‍💻 Author

Audio processing pipeline using Librosa + Matplotlib
