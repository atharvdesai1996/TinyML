"""
Audio Preprocessing for Keyword Spotting: Spectrograms and MFCC Analysis
========================================================================
This script demonstrates audio signal processing fundamentals for ML,
particularly keyword spotting (wake word detection).

CONCEPTS: Audio Recording | Time Domain | FFT | Spectrograms | MFCC
STAGE: Data Collection and Feature Engineering
"""

import subprocess, sys

# Install packages (Google Colab)
print("Installing packages...")
for pkg in ["ffmpeg-python", "tensorflow-io", "python_speech_features", "librosa"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(" Packages Installed")

# Imports
from IPython.display import HTML, Audio
from google.colab.output import eval_js
from base64 import b64decode
import numpy as np
from scipy.io.wavfile import read as wav_read
import io, ffmpeg, pickle
import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt
from matplotlib import cm
import librosa

print("✓ Packages Imported")

# ==========================================
# BROWSER-BASED AUDIO RECORDER
# ==========================================
"""
WHY: Works in Colab without hardware setup
HOW: JavaScript MediaRecorder API → WebM → WAV conversion
USE: Custom wake word/keyword data collection
"""

AUDIO_HTML = """
<script>
var my_div = document.createElement("DIV");
var my_btn = document.createElement("BUTTON");
my_btn.appendChild(document.createTextNode("Press to start recording"));
my_div.appendChild(my_btn);
document.body.appendChild(my_div);

var base64data = 0, reader, recorder, gumStream;
var recordButton = my_btn;

var handleSuccess = function(stream) {
  gumStream = stream;
  recorder = new MediaRecorder(stream, {mimeType: 'audio/webm;codecs=opus'});
  recorder.ondataavailable = function(e) {
    var url = URL.createObjectURL(e.data);
    var preview = document.createElement('audio');
    preview.controls = true;
    preview.src = url;
    document.body.appendChild(preview);
    reader = new FileReader();
    reader.readAsDataURL(e.data);
    reader.onloadend = function() { base64data = reader.result; }
  };
  recorder.start();
};

recordButton.innerText = "Recording... press to stop";
navigator.mediaDevices.getUserMedia({audio: true}).then(handleSuccess);

function toggleRecording() {
  if (recorder && recorder.state == "recording") {
      recorder.stop();
      gumStream.getAudioTracks()[0].stop();
      recordButton.innerText = "Saving... pls wait!"
  }
}

function sleep(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }

var data = new Promise(resolve=>{
  recordButton.onclick = ()=>{
    toggleRecording()
    sleep(2000).then(() => { resolve(base64data.toString()) });
  }
});
</script>
"""

def get_audio():
    """Capture audio from browser, convert WebM to WAV, return numpy array"""
    display(HTML(AUDIO_HTML))
    data = eval_js("data")
    binary = b64decode(data.split(',')[1])
    
    # WebM → WAV conversion via ffmpeg
    process = (ffmpeg.input('pipe:0').output('pipe:1', format='wav')
               .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, 
                         quiet=True, overwrite_output=True))
    output, err = process.communicate(input=binary)
    
    # Fix WAV header (RIFF chunk size)
    riff_chunk_size = len(output) - 8
    q, b = riff_chunk_size, []
    for i in range(4):
        q, r = divmod(q, 256)
        b.append(r)
    riff = output[:4] + bytes(b) + output[8:]
    
    sr, audio = wav_read(io.BytesIO(riff))
    return audio, sr

print("✓ Audio Recorder Ready")
print("\n" + "="*60)
print("INSTRUCTIONS: Record 4 samples (yes-loud, yes-quiet, no-loud, no-quiet)")
print("="*60)

# ==========================================
# RECORD AUDIO SAMPLES
# ==========================================
"""
DATA COLLECTION: 4 variations for robustness testing
- Tests model generalization across volume levels
- Identifies volume-invariant features
"""

print("\n 1/4: Say 'YES' LOUDLY")
audio_yes_loud, sr_yes_loud = get_audio()

print(" 2/4: Say 'yes' quietly")
audio_yes_quiet, sr_yes_quiet = get_audio()

print(" 3/4: Say 'NO' LOUDLY")
audio_no_loud, sr_no_loud = get_audio()

print(" 4/4: Say 'no' quietly")
audio_no_quiet, sr_no_quiet = get_audio()

print("✓ Recording Complete!")

# Save for reuse
audio_files = {
    'audio_yes_loud': audio_yes_loud, 'sr_yes_loud': sr_yes_loud,
    'audio_yes_quiet': audio_yes_quiet, 'sr_yes_quiet': sr_yes_quiet,
    'audio_no_loud': audio_no_loud, 'sr_no_loud': sr_no_loud,
    'audio_no_quiet': audio_no_quiet, 'sr_no_quiet': sr_no_quiet,
}
with open('audio_files.pkl', 'wb') as fid:
    pickle.dump(audio_files, fid)
print("✓ Saved to audio_files.pkl")

# ==========================================
# VISUALIZATION 1: TIME DOMAIN (Waveforms)
# ==========================================
"""
TIME DOMAIN: Raw amplitude over time
SHOWS: Volume differences, temporal patterns
ISSUE: Not volume-invariant (problem for ML!)
"""
print("\n" + "="*60)
print("VIZ 1: TIME DOMAIN (Raw Waveforms)")
print("="*60)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
max_val = max(np.concatenate([audio_yes_loud, audio_yes_quiet, audio_no_loud, audio_no_quiet]))

for ax, audio, title in [(ax1, audio_yes_loud, "Yes Loud"),
                          (ax2, audio_yes_quiet, "Yes Quiet"),
                          (ax3, audio_no_loud, "No Loud"),
                          (ax4, audio_no_quiet, "No Quiet")]:
    ax.plot(audio)
    ax.set_title(title, fontsize=20, fontweight='bold')
    ax.set_ylim(-max_val, max_val)
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Sample')

fig.set_size_inches(18, 12)
plt.tight_layout()
plt.savefig('time_domain_waveforms.png', dpi=150)
plt.show()
print("✓ Saved: time_domain_waveforms.png")

# ==========================================
# VISUALIZATION 2: FREQUENCY DOMAIN (FFT)
# ==========================================
"""
FFT: Converts time → frequency domain
SHOWS: Which frequencies present, harmonic structure
BETTER: More discriminative than raw waveform for speech
KEY: Human speech = 80-8000 Hz, formants identify vowels
"""
print("\n" + "="*60)
print("VIZ 2: FREQUENCY DOMAIN (FFT)")
print("="*60)

# Compute FFT (magnitude only, positive frequencies)
fft_data = [(np.abs(2*np.fft.fft(audio_yes_loud)), "Yes Loud"),
            (np.abs(2*np.fft.fft(audio_yes_quiet)), "Yes Quiet"),
            (np.abs(2*np.fft.fft(audio_no_loud)), "No Loud"),
            (np.abs(2*np.fft.fft(audio_no_quiet)), "No Quiet")]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
for ax, (ft, title) in zip([ax1, ax2, ax3, ax4], fft_data):
    ax.plot(ft)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(title, fontsize=20, fontweight='bold')

fig.set_size_inches(18, 12)
fig.text(0.5, 0.06, 'Frequency [Hz]', fontsize=20, fontweight='bold', ha='center')
fig.text(0.08, 0.5, 'Amplitude', fontsize=20, fontweight='bold', va='center', rotation=90)
plt.tight_layout()
plt.savefig('frequency_domain_fft.png', dpi=150)
plt.show()
print("✓ Saved: frequency_domain_fft.png")

# ==========================================
# VISUALIZATION 3: SPECTROGRAMS
# ==========================================
"""
SPECTROGRAM: Time-frequency representation (2D image)
AXES: X=time, Y=frequency, Color=energy
HOW: Short-time FFT on overlapping windows
USE: CNN input, shows formant transitions
PARAMS: nfft=freq resolution, stride=time resolution
"""
print("\n" + "="*60)
print("VIZ 3: SPECTROGRAMS (Time-Frequency)")
print("="*60)

# Compute spectrograms
specs = [(tfio.audio.spectrogram(audio/1.0, nfft=2048, window=len(audio), 
                                  stride=int(sr*0.008)), title)
         for audio, sr, title in [(audio_yes_loud, sr_yes_loud, "Yes Loud"),
                                   (audio_yes_quiet, sr_yes_quiet, "Yes Quiet"),
                                   (audio_no_loud, sr_no_loud, "No Loud"),
                                   (audio_no_quiet, sr_no_quiet, "No Quiet")]]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
for ax, (spec, title) in zip([ax1, ax2, ax3, ax4], specs):
    ax.imshow(tf.math.log(spec).numpy(), aspect='auto', cmap='viridis')
    ax.set_title(title, fontsize=20, fontweight='bold')
    ax.set_ylabel('Frequency Bin')
    ax.set_xlabel('Time Frame')

fig.set_size_inches(18, 12)
plt.tight_layout()
plt.savefig('spectrograms.png', dpi=150)
plt.show()
print("✓ Saved: spectrograms.png")

# ==========================================
# VISUALIZATION 4: MFCC (MEL-FREQUENCY)
# ==========================================
"""
MFCC: Gold standard for speech ML
WHY BEST:
  ✓ Perceptual (mimics human hearing via Mel scale)
  ✓ Compact (13-40 coefficients vs 1000s of bins)
  ✓ Robust to noise (cepstral processing)
  ✓ Volume-invariant (spectral shape, not power)

PROCESS: Spectrogram → Mel filterbank → Log → DCT → MFCCs
PARAMS: n_mels=128 Mel bands, n_fft/hop_length control resolution
"""
print("\n" + "="*60)
print("VIZ 4: MFCC (Mel-Frequency Cepstral Coefficients)")
print("="*60)

# Compute MFCCs via Mel spectrogram
mfccs = []
for audio, sr, title in [(audio_yes_loud, sr_yes_loud, "Yes Loud"),
                         (audio_yes_quiet, sr_yes_quiet, "Yes Quiet"),
                         (audio_no_loud, sr_no_loud, "No Loud"),
                         (audio_no_quiet, sr_no_quiet, "No Quiet")]:
    mfcc_data = librosa.power_to_db(
        librosa.feature.melspectrogram(y=np.float32(audio), sr=sr, 
                                       n_fft=2048, hop_length=512, n_mels=128),
        ref=np.max)
    mfccs.append((mfcc_data, title))

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
for ax, (mfcc_data, title) in zip([ax1, ax2, ax3, ax4], mfccs):
    ax.imshow(np.swapaxes(mfcc_data, 0, 1), interpolation='nearest',
              cmap=cm.viridis, origin='lower', aspect='auto')
    ax.set_title(title, fontsize=20, fontweight='bold')
    ax.set_ylabel('Mel Frequency Band')
    ax.set_xlabel('Time Frame')
    ax.set_ylim(ax.get_ylim()[::-1])

fig.set_size_inches(18, 12)
plt.tight_layout()
plt.savefig('mfcc_analysis.png', dpi=150)
plt.show()
print("✓ Saved: mfcc_analysis.png")

# ==========================================
# SUMMARY
# ==========================================
print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print("""
 REPRESENTATION COMPARISON:

1. TIME DOMAIN (Waveform)
   - Shows amplitude over time
   - Volume-dependent 
   - Not ideal for ML

2. FREQUENCY DOMAIN (FFT)
   - Shows frequency content
   - Better than waveform
   - Loses temporal info 

3. SPECTROGRAM
   - Time AND frequency ✓
   - CNN-compatible (2D image)
   - Still volume-sensitive 

4. MFCC (RECOMMENDED) 
   ✓ Perceptually motivated
   ✓ Compact (13-40 features)
   ✓ Volume-invariant
   ✓ Noise-robust
   ✓ Industry standard

📁 FILES GENERATED:
- audio_files.pkl
- time_domain_waveforms.png
- frequency_domain_fft.png
- spectrograms.png
- mfcc_analysis.png
""")
print("="*60)
print("Ready for keyword spotting model!")
print("="*60)
