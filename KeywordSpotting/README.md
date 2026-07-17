# Keyword Spotting - Audio ML for TinyML

This directory contains assignments and tutorials for building keyword spotting (wake word detection) systems that can run on edge devices.

---

##  What is Keyword Spotting?

**Keyword Spotting** (also called wake word detection) is the task of identifying specific words or phrases in an audio stream. Common examples:
- "Hey Siri" (Apple)
- "OK Google" (Google)
- "Alexa" (Amazon)
- Custom wake words for IoT devices

### Why TinyML for Keyword Spotting?
- **Privacy**: Processing on-device (no cloud needed)
- **Low Latency**: Instant response (<100ms)
- **Always-On**: Minimal power consumption
- **Offline**: Works without internet connectivity

---

## Contents

- **`audio_preprocessing_spectrograms_mfcc.py`** - Introduction to audio feature extraction
- **Generated visualizations** (from running the script):
  - `time_domain_waveforms.png`
  - `frequency_domain_fft.png`
  - `spectrograms.png`
  - `mfcc_analysis.png`
  - `audio_files.pkl`

---

## Audio Preprocessing Assignment

### Overview
This assignment introduces the foundational concepts of audio signal processing for machine learning. Before building a keyword spotting model, you must understand how audio is represented and which features work best for ML.

### Learning Objectives
1. **Understand Audio Representations**:
   - Time domain (waveforms)
   - Frequency domain (FFT)
   - Time-frequency domain (spectrograms)
   - Perceptual domain (MFCC)

2. **Data Collection**:
   - Record audio samples in controlled conditions
   - Test robustness across volume variations
   - Create reusable datasets

3. **Feature Engineering**:
   - Extract meaningful features from raw audio
   - Understand why MFCC is optimal for speech
   - Prepare data for ML model training

---

##  Audio Representations Explained

### 1. Time Domain (Waveform)

![Time Domain Example](time_domain_waveforms.png)

**What it shows**: Raw audio amplitude over time

**Characteristics**:
- X-axis: Time (samples)
- Y-axis: Amplitude (loudness)
- Direct microphone output


**Use Case**: Initial data inspection, quality checking

---

### 2. Frequency Domain (FFT)

![Frequency Domain Example](frequency_domain_fft.png)

**What it shows**: Which frequencies are present in the audio

**How it's computed**: Fast Fourier Transform (FFT) converts time → frequency

**Characteristics**:
- X-axis: Frequency (Hz)
- Y-axis: Magnitude/Amplitude
- Shows harmonic structure

**Key Insights**:
- Human speech: 80-8000 Hz
- Fundamental frequency (pitch): 80-250 Hz
- Formants (vowel characteristics): 500-3500 Hz

**Pros**:
- Reveals frequency components
-  More discriminative than waveform
- Shows phonetic characteristics

**Cons**:
-  Loses all temporal information
-  Can't see how sound evolves over time
-  Single snapshot, not continuous

**Use Case**: Understanding frequency content, spectral analysis

---

### 3. Spectrogram (Time-Frequency)

![Spectrogram Example](spectrograms.png)

**What it shows**: How frequency content changes over time (2D image)

**How it's computed**:
1. Divide audio into short windows (e.g., 25ms)
2. Apply FFT to each window
3. Stack results to create 2D representation

**Characteristics**:
- X-axis: Time (frames)
- Y-axis: Frequency (bins)
- Color/Intensity: Energy at that time-frequency point

**Parameters**:
- `nfft=2048`: FFT size (frequency resolution)
- `window`: Window length
- `stride`: Hop length between windows (time resolution)

**Pros**:
-  Captures both time AND frequency
-  Can be fed to CNNs as 2D images
-  Shows formant transitions (key for speech)
-  Visualizes temporal dynamics

**Cons**:
-  Still somewhat volume-sensitive
-  High dimensional (1025 freq bins × time frames)
-  Not perceptually motivated

**Use Case**: CNN input, visual inspection, phoneme analysis

---

### 4. MFCC (Mel-Frequency Cepstral Coefficients) 

![MFCC Example](mfcc_analysis.png)

**What it shows**: Perceptually-motivated compact representation of audio

**Why MFCC is the GOLD STANDARD for Speech**:

####  **Perceptually Motivated**
- Based on **Mel scale** (mimics human hearing)
- More resolution at low frequencies (where speech energy concentrates)
- Less resolution at high frequencies (less perceptually important)
- Formula: `Mel(f) = 2595 * log10(1 + f/700)`

####  **Extreme Dimensionality Reduction**
- Typical spectrogram: **1025 frequency bins**
- MFCC: **13-40 coefficients** (50x+ reduction!)
- Compresses spectral envelope (formant structure)
- Retains essential information

####  **Volume Invariant**
- Focuses on **spectral shape**, not absolute power
- Better generalization across recording conditions
- Robust to microphone placement and distance
- Ideal for real-world deployment

####  **Noise Robust**
- Cepstral processing separates:
  - **Source**: Vocal cord vibrations (pitch)
  - **Filter**: Vocal tract shape (phonemes)
- Emphasizes formants over noise
- Better SNR (Signal-to-Noise Ratio) handling


**How MFCCs are Computed**:
```
Raw Audio
  ↓
Spectrogram (STFT)
  ↓
Mel Filterbank (perceptual frequency warping)
  ↓
Logarithm (compress dynamic range)
  ↓
DCT (Discrete Cosine Transform)
  ↓
MFCCs (first 13-40 coefficients)
```

**Parameters**:
- `n_fft=2048`: FFT window size
- `hop_length=512`: Samples between frames
- `n_mels=128`: Number of Mel frequency bands
- Typical output: 13-40 MFCC coefficients


**Use Case**: **Primary input for all speech ML models** (RNN, CNN, DNN)

---

### Step 1: Data Collection
```python
# Record 4 variations:
# 1. "YES" loud
# 2. "yes" quiet
# 3. "NO" loud
# 4. "no" quiet
```

**Why 4 variations?**
- Tests model robustness to volume changes
- Identifies volume-invariant features
- Simulates real-world recording conditions

### Step 2: Time Domain Analysis
- Visualize raw waveforms
- Observe amplitude differences between loud/quiet
- Understand why waveforms aren't ideal for ML

### Step 3: Frequency Domain (FFT)
- Convert to frequency representation
- Identify frequency peaks (formants)
- Compare "yes" vs "no" frequency patterns

### Step 4: Spectrogram Generation
- Create time-frequency 2D images
- See how sound evolves over time
- Identify formant transitions

### Step 5: MFCC Extraction 
- Generate Mel-frequency cepstral coefficients
- Observe volume invariance (loud/quiet look similar!)
- Prepare features for ML training

---

---

## Key Takeaways

1.  How audio is represented digitally
2.  Different ways to transform audio for ML
3.  Why MFCC is optimal for speech recognition
4.  How to collect and preprocess audio data
5.  The importance of volume invariance

### Why It Matters:
- **Foundation for all audio ML**: Understanding these concepts is crucial
- **Feature engineering**: Good features = better models
- **Real-world robustness**: MFCC handles varying recording conditions
- **Efficient deployment**: Compact features fit on microcontrollers

---

##  Next Steps

### 1. Scale Up Data Collection
- Record 100+ samples per keyword
- Add more keywords (yes, no, go, stop, left, right, etc.)
- Include background noise variations
- Collect from multiple speakers

### 2. Build ML Model
```python
# Typical architecture:
Input (MFCC features)
  ↓
CNN/RNN layers
  ↓
Dense layers
  ↓
Softmax (keyword classes)
```

### 3. Convert to TFLite
- Apply quantization (see TFlite section)
- Optimize for edge devices
- Target <50KB model size

### 4. Deploy on Microcontroller
- Arduino Nano 33 BLE Sense
- ESP32 with I2S microphone
- STM32 with MEMS mic
- Raspberry Pi Pico

