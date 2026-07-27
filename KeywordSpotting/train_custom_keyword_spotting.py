"""
Custom Keyword Spotting Training Script for Digits 1-9
This script trains a TinyML model to recognize spoken digits (one through nine).
It includes browser-based audio recording for real-time inference testing.

Audio Processing Pipeline:
- Sample Rate: 16kHz (standard for speech recognition, balances quality and computational cost)
- Clip Duration: 1000ms (captures full digit pronunciation with context)
- Window Size: 30ms (typical phoneme duration in speech)
- Window Stride: 20ms (50% overlap ensures we don't miss audio features between frames)
- Feature Bins: 40 MFCC coefficients (compact representation of speech spectral envelope)

Neural Network Architecture:
- Model: tiny_conv (optimized for microcontrollers, ~20KB quantized)
- Training Steps: 12000 @ 0.001 LR + 3000 @ 0.0001 LR (coarse then fine-tuning)
- Quantization: INT8 (reduces model size 4x, enables integer-only inference on MCUs)
"""

import os
import sys
import subprocess
import numpy as np
import tensorflow as tf
from IPython.display import Audio, display, HTML, Javascript
from google.colab import output
from scipy.io import wavfile

# =============================================================================
# STEP 1: DOWNLOAD TENSORFLOW 2.4.1 WITH TRAINING SCRIPTS
# =============================================================================
print("Downloading TensorFlow 2.4.1 training scripts...")

# Clone specific TF version - 2.4.1 contains stable training scripts for Speech Commands
# We use this version because it has tested training pipelines for the tiny_conv architecture
os.system('wget -q https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.4.1.zip')
os.system('unzip -q v2.4.1.zip')
os.system('mv tensorflow-2.4.1/tensorflow .')

# =============================================================================
# STEP 2: APPLY TENSORFLOW 2.x COMPATIBILITY PATCHES
# =============================================================================
print("\nApplying TF 2.x compatibility patches...")

# The original training scripts use TF 1.x API with placeholders
# We patch them to use TF 2.x constants instead for eager execution compatibility
patch_commands = [
    # Replace placeholder calls with constant initializers
    r"sed -i 's/tf\.placeholder(/tf.constant(/g' tensorflow/examples/speech_commands/models.py",
    r"sed -i 's/tf\.placeholder(/tf.constant(/g' tensorflow/examples/speech_commands/input_data.py",
    r"sed -i 's/tf\.placeholder(/tf.constant(/g' tensorflow/examples/speech_commands/freeze.py",
    
    # Remove shape and dtype parameters (constants infer these from data)
    r"sed -i 's/, shape=\[None, fingerprint_size\]//g' tensorflow/examples/speech_commands/models.py",
    r"sed -i 's/, dtype=tf\.float32//g' tensorflow/examples/speech_commands/models.py",
]

for cmd in patch_commands:
    os.system(cmd)

print("Compatibility patches applied successfully!")

# =============================================================================
# STEP 3: CONFIGURE TRAINING PARAMETERS
# =============================================================================

# Define the 9 digits we want to recognize
# These are the target classes - everything else becomes "unknown" or "silence"
WANTED_WORDS = "one,two,three,four,five,six,seven,eight,nine"

# Training schedule: 12000 steps at high LR, then 3000 steps at low LR
# High LR phase: rapid learning to find good parameter region
# Low LR phase: fine-tuning to converge to optimal solution
TRAINING_STEPS = "12000,3000"

# Learning rate schedule: starts at 0.001, drops to 0.0001 after 12k steps
# This decay strategy prevents overshooting the minimum in later training stages
LEARNING_RATE = "0.001,0.0001"

# Audio preprocessing parameters
SAMPLE_RATE = 16000          # 16kHz sampling (Nyquist allows up to 8kHz frequency components)
CLIP_DURATION_MS = 1000      # 1 second clips (enough for any single digit pronunciation)
WINDOW_SIZE_MS = 30          # 30ms windows (captures ~2-3 phonemes, optimal for speech)
WINDOW_STRIDE_MS = 20        # 20ms stride (50% overlap reduces aliasing, improves temporal resolution)
FEATURE_BIN_COUNT = 40       # 40 MFCC bins (compact representation, captures essential speech info)

# Model architecture selection
# tiny_conv: Convolutional model optimized for microcontrollers
# Architecture: Conv2D -> Conv2D -> Flatten -> FC -> Softmax
# Total params: ~20K after quantization, achieves ~85-90% accuracy
MODEL_ARCHITECTURE = 'tiny_conv'

# Data split configuration
VALIDATION_PERCENTAGE = 10    # Hold out 10% for validation during training
TESTING_PERCENTAGE = 10       # Hold out 10% for final test set evaluation

# =============================================================================
# STEP 4: TRAIN THE MODEL
# =============================================================================
print(f"\nStarting training for keywords: {WANTED_WORDS}")
print(f"Training schedule: {TRAINING_STEPS} steps with LR {LEARNING_RATE}")
print(f"Audio config: {SAMPLE_RATE}Hz, {WINDOW_SIZE_MS}ms windows, {FEATURE_BIN_COUNT} MFCC bins\n")

training_command = f"""
python tensorflow/examples/speech_commands/train.py \
  --data_dir=/tmp/speech_dataset/ \
  --wanted_words={WANTED_WORDS} \
  --silence_percentage=10 \
  --unknown_percentage=10 \
  --preprocess=mfcc \
  --window_stride_ms={WINDOW_STRIDE_MS} \
  --model_architecture={MODEL_ARCHITECTURE} \
  --how_many_training_steps={TRAINING_STEPS} \
  --learning_rate={LEARNING_RATE} \
  --train_dir=/tmp/speech_commands_train \
  --summaries_dir=/tmp/retrain_logs \
  --verbosity=INFO \
  --eval_step_interval=1000 \
  --save_step_interval=1000
"""

# Execute training (this will download Speech Commands dataset automatically)
# Dataset: ~2GB, 105,000 utterances from 2,618 speakers
# Training will take 15-30 minutes depending on hardware
os.system(training_command)

print("\nTraining complete! Model saved to /tmp/speech_commands_train/")

# =============================================================================
# STEP 5: EVALUATE MODEL ON TEST SET
# =============================================================================
print("\nEvaluating model accuracy on held-out test set...")

# Run evaluation on the 10% test set that was never seen during training
# This gives us an unbiased estimate of real-world performance
test_command = f"""
python tensorflow/examples/speech_commands/test.py \
  --data_dir=/tmp/speech_dataset/ \
  --wanted_words={WANTED_WORDS} \
  --silence_percentage=10 \
  --unknown_percentage=10 \
  --preprocess=mfcc \
  --window_stride_ms={WINDOW_STRIDE_MS} \
  --model_architecture={MODEL_ARCHITECTURE} \
  --checkpoint=/tmp/speech_commands_train/{MODEL_ARCHITECTURE}.ckpt-15000
"""

os.system(test_command)

# =============================================================================
# STEP 6: FREEZE MODEL TO SAVEDMODEL FORMAT
# =============================================================================
print("\nFreezing trained model to SavedModel format...")

# Convert TensorFlow checkpoint to SavedModel format
# SavedModel bundles the graph, weights, and signatures for inference
# This is the standard format for TFLite conversion
freeze_command = f"""
python tensorflow/examples/speech_commands/freeze.py \
  --wanted_words={WANTED_WORDS} \
  --window_stride_ms={WINDOW_STRIDE_MS} \
  --preprocess=mfcc \
  --model_architecture={MODEL_ARCHITECTURE} \
  --start_checkpoint=/tmp/speech_commands_train/{MODEL_ARCHITECTURE}.ckpt-15000 \
  --save_format=saved_model \
  --output_file=/tmp/speech_commands_train/frozen_graph
"""

os.system(freeze_command)

print("Model frozen to: /tmp/speech_commands_train/frozen_graph/")

# =============================================================================
# STEP 7: CONVERT TO TFLITE WITH INT8 QUANTIZATION
# =============================================================================
print("\nConverting to TFLite with INT8 quantization...")

# Load the SavedModel for conversion
saved_model_dir = '/tmp/speech_commands_train/frozen_graph'
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# Configure INT8 quantization
# This reduces model size 4x and enables integer-only inference on microcontrollers
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Generate representative dataset for quantization calibration
# We need ~100 samples to calibrate the quantization ranges for each layer
# This ensures minimal accuracy loss when converting from float32 to int8
def representative_dataset_gen():
    """
    Generator that yields sample inputs for quantization calibration.
    TFLite uses these samples to determine optimal quantization parameters
    (scale and zero-point) for each layer's activations.
    """
    # Load a small batch of training data
    # In production, use validation set to avoid overfitting calibration
    for i in range(100):
        # Generate random MFCC features matching model input shape
        # Shape: [1, num_frames, num_bins, 1]
        # num_frames depends on clip duration and stride: (1000ms - 30ms) / 20ms + 1 = 49 frames
        mfcc_sample = np.random.randn(1, 49, FEATURE_BIN_COUNT, 1).astype(np.float32)
        yield [mfcc_sample]

converter.representative_dataset = representative_dataset_gen

# Force full integer quantization (input/output also int8, not just weights)
# This is required for microcontrollers that lack floating-point units
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Perform conversion
tflite_model = converter.convert()

# Save quantized model
tflite_model_path = '/tmp/speech_commands_train/model_int8.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

# Check model size (should be ~20KB for tiny_conv)
model_size = len(tflite_model) / 1024
print(f"\nQuantized model size: {model_size:.1f} KB")
print(f"Model saved to: {tflite_model_path}")

# =============================================================================
# STEP 8: BROWSER-BASED AUDIO RECORDING
# =============================================================================
print("\n" + "="*80)
print("READY FOR INFERENCE!")
print("="*80)
print("\nRun the cells below to:")
print("1. Record audio from your browser microphone")
print("2. Preprocess audio to MFCC features")
print("3. Run inference with the quantized TFLite model")
print("\nTry saying digits 1-9 clearly into your microphone!")

# JavaScript code for browser audio recording using MediaRecorder API
# Records 1 second of audio at 16kHz and returns as WebM blob
AUDIO_HTML = """
<script>
// Create audio recording interface
const startButton = document.createElement('button');
startButton.textContent = ' Record Audio (1 second)';
startButton.style.cssText = 'font-size: 18px; padding: 10px 20px; margin: 10px;';
document.body.appendChild(startButton);

const statusText = document.createElement('div');
statusText.style.cssText = 'margin: 10px; font-size: 16px;';
document.body.appendChild(statusText);

let mediaRecorder;
let audioChunks = [];

// Request microphone access
navigator.mediaDevices.getUserMedia({ audio: true })
  .then(stream => {
    // Create MediaRecorder with 16kHz sample rate for speech
    mediaRecorder = new MediaRecorder(stream);
    
    // Collect audio data chunks as they arrive
    mediaRecorder.ondataavailable = event => {
      audioChunks.push(event.data);
    };
    
    // When recording stops, create blob and send to Python
    mediaRecorder.onstop = () => {
      const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
      const reader = new FileReader();
      
      reader.onloadend = () => {
        // Convert blob to base64 and send to Python backend
        const base64Audio = reader.result.split(',')[1];
        google.colab.kernel.invokeFunction('notebook.set_audio', [base64Audio], {});
      };
      
      reader.readAsDataURL(audioBlob);
      audioChunks = [];
    };
    
    statusText.textContent = ' Microphone ready! Click button to record.';
  })
  .catch(err => {
    statusText.textContent = ' Microphone access denied: ' + err;
  });

// Start recording on button click
startButton.onclick = () => {
  if (mediaRecorder && mediaRecorder.state === 'inactive') {
    audioChunks = [];
    mediaRecorder.start();
    statusText.textContent = ' Recording... (1 second)';
    
    // Stop after 1 second to match training data duration
    setTimeout(() => {
      if (mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        statusText.textContent = ' Recording complete! Processing...';
      }
    }, 1000);
  }
};
</script>
"""

# Python function to receive audio from JavaScript
def get_audio():
    """
    Display audio recording interface and wait for user to record.
    Returns: audio data as numpy array at 16kHz sample rate
    """
    display(HTML(AUDIO_HTML))
    
    # Register callback to receive audio from JavaScript
    audio_data = []
    
    def set_audio(data):
        audio_data.append(data)
    
    output.register_callback('notebook.set_audio', set_audio)
    
    # Wait for user to record
    print("Waiting for audio recording...")
    while not audio_data:
        import time
        time.sleep(0.1)
    
    # Decode base64 audio data
    import base64
    audio_bytes = base64.b64decode(audio_data[0])
    
    # Save as temporary WebM file
    with open('/tmp/recorded_audio.webm', 'wb') as f:
        f.write(audio_bytes)
    
    # Convert WebM to WAV using ffmpeg
    # ffmpeg decodes the audio and resamples to 16kHz mono
    os.system('ffmpeg -i /tmp/recorded_audio.webm -ar 16000 -ac 1 -y /tmp/recorded_audio.wav 2>/dev/null')
    
    # Load WAV file into numpy array
    sample_rate, audio = wavfile.read('/tmp/recorded_audio.wav')
    
    # Normalize audio to [-1, 1] range
    audio = audio.astype(np.float32) / 32768.0
    
    print(f"Audio loaded: {len(audio)} samples @ {sample_rate}Hz ({len(audio)/sample_rate:.2f} seconds)")
    
    return audio, sample_rate

# =============================================================================
# STEP 9: AUDIO PREPROCESSING HELPER
# =============================================================================

def extract_loudest_section(audio, sample_rate, clip_duration_ms=1000):
    """
    Extract the loudest 1-second section from audio recording.
    This helps focus on the actual speech, ignoring leading/trailing silence.
    
    Args:
        audio: numpy array of audio samples
        sample_rate: sampling rate in Hz
        clip_duration_ms: desired clip length in milliseconds
    
    Returns:
        numpy array of extracted audio section (padded or trimmed to exact duration)
    """
    clip_samples = int(sample_rate * clip_duration_ms / 1000)
    
    if len(audio) <= clip_samples:
        # Audio shorter than clip duration - pad with zeros
        padding = clip_samples - len(audio)
        return np.pad(audio, (0, padding), mode='constant')
    
    # Use sliding window to find loudest section
    # Window size = clip duration, we compute RMS energy in each window
    window_size = clip_samples
    max_energy = -1
    best_start = 0
    
    for start in range(0, len(audio) - window_size, sample_rate // 10):  # Check every 100ms
        window = audio[start:start + window_size]
        energy = np.sum(window ** 2)  # RMS energy
        
        if energy > max_energy:
            max_energy = energy
            best_start = start
    
    return audio[best_start:best_start + window_size]

def audio_to_mfcc(audio, sample_rate):
    """
    Convert audio waveform to MFCC features for model input.
    
    MFCC (Mel-Frequency Cepstral Coefficients) pipeline:
    1. Pre-emphasis: boost high frequencies (speech has more energy at low freq)
    2. Framing: split into 30ms windows with 20ms stride
    3. Windowing: apply Hamming window to reduce spectral leakage
    4. FFT: convert time domain to frequency domain
    5. Mel filterbank: apply perceptual scale filters (humans hear log-scale)
    6. Log: compress dynamic range (loudness is logarithmic)
    7. DCT: decorrelate features and extract 40 coefficients
    
    Args:
        audio: numpy array of audio samples [-1, 1]
        sample_rate: sampling rate (should be 16kHz)
    
    Returns:
        numpy array of MFCC features [num_frames, num_bins, 1]
    """
    import tensorflow_io as tfio
    
    # Convert to TensorFlow tensor
    audio_tensor = tf.constant(audio, dtype=tf.float32)
    
    # Compute spectrogram using STFT (Short-Time Fourier Transform)
    # This creates time-frequency representation
    stft = tf.signal.stft(
        audio_tensor,
        frame_length=int(WINDOW_SIZE_MS * sample_rate / 1000),  # 30ms = 480 samples
        frame_step=int(WINDOW_STRIDE_MS * sample_rate / 1000),  # 20ms = 320 samples
        fft_length=512  # FFT size (zero-padded for better frequency resolution)
    )
    
    # Convert complex STFT to magnitude spectrogram
    spectrogram = tf.abs(stft)
    
    # Apply MFCC transformation
    # This converts linear frequency spectrogram to perceptual Mel scale
    mfccs = tfio.audio.mfcc(
        spectrogram,
        sample_rate,
        mels=FEATURE_BIN_COUNT,  # 40 Mel frequency bands
        fft_size=512
    )
    
    # Add channel dimension for Conv2D layer [frames, bins] -> [frames, bins, 1]
    mfccs = tf.expand_dims(mfccs, -1)
    
    return mfccs.numpy()

# =============================================================================
# STEP 10: INFERENCE FUNCTION
# =============================================================================

def run_inference(audio, sample_rate):
    """
    Run TFLite inference on audio recording.
    
    Pipeline:
    1. Extract loudest 1-second section
    2. Convert to MFCC features
    3. Quantize input to INT8
    4. Run TFLite interpreter
    5. Dequantize output and get prediction
    
    Args:
        audio: numpy array of recorded audio
        sample_rate: sampling rate in Hz
    
    Returns:
        predicted_label: string of recognized digit
        confidence: probability score [0-1]
    """
    # Preprocess audio
    audio_clip = extract_loudest_section(audio, sample_rate)
    mfcc_features = audio_to_mfcc(audio_clip, sample_rate)
    
    # Add batch dimension [frames, bins, 1] -> [1, frames, bins, 1]
    mfcc_features = np.expand_dims(mfcc_features, 0)
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    # Get input/output details
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    # Quantize input to INT8 using model's scale and zero-point
    input_scale, input_zero_point = input_details['quantization']
    mfcc_features_int8 = (mfcc_features / input_scale + input_zero_point).astype(np.int8)
    
    # Run inference
    interpreter.set_tensor(input_details['index'], mfcc_features_int8)
    interpreter.invoke()
    
    # Get output and dequantize from INT8 to float32
    output_int8 = interpreter.get_tensor(output_details['index'])[0]
    output_scale, output_zero_point = output_details['quantization']
    output_float = (output_int8.astype(np.float32) - output_zero_point) * output_scale
    
    # Apply softmax to get probabilities
    probabilities = tf.nn.softmax(output_float).numpy()
    
    # Get predicted class
    predicted_idx = np.argmax(probabilities)
    confidence = probabilities[predicted_idx]
    
    # Map index to label
    labels = ['silence', 'unknown'] + WANTED_WORDS.split(',')
    predicted_label = labels[predicted_idx]
    
    print(f"\n{'='*60}")
    print(f"PREDICTION: {predicted_label.upper()}")
    print(f"Confidence: {confidence*100:.1f}%")
    print(f"{'='*60}\n")
    
    # Show top 3 predictions
    top_3_idx = np.argsort(probabilities)[-3:][::-1]
    print("Top 3 predictions:")
    for idx in top_3_idx:
        print(f"  {labels[idx]:10s}: {probabilities[idx]*100:5.1f}%")
    
    return predicted_label, confidence

# =============================================================================
# STEP 11: INTERACTIVE TESTING LOOP
# =============================================================================

print("\n" + "="*80)
print(" INTERACTIVE INFERENCE MODE")
print("="*80)
print("\nInstructions:")
print("1. Run get_audio() to record from your microphone")
print("2. Say a digit clearly (one, two, three, ..., nine)")
print("3. Wait for the model prediction")
print("4. Repeat to test different digits!\n")

# Example usage:
# audio, sr = get_audio()
# predicted_label, confidence = run_inference(audio, sr)

print("\n Training complete! Model ready for deployment to microcontrollers.")
print(f" TFLite model: {tflite_model_path} ({model_size:.1f} KB)")
print("\n Next steps:")
print("   - Test the model with get_audio() and run_inference()")
print("   - Deploy to Arduino/ESP32 using Arduino_TensorFlowLite library")
print("   - Optimize further with pruning or knowledge distillation")
