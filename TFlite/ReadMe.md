# TensorFlow Lite (TFLite) - Model Optimization and Deployment

This directory contains scripts and examples for training, optimizing, and deploying TensorFlow models using TensorFlow Lite for edge devices and mobile platforms.

---

## Contents

- **`train_and_convert_to_tflite.py`** - Production-ready script for training and converting models to TFLite
- **`quantization_optimization_assignment.py`** - Comprehensive assignment on model quantization techniques
- **`tflite_models/`** - Directory containing converted TFLite models

---

##  What is TensorFlow Lite?

TensorFlow Lite is a lightweight solution for running machine learning models on mobile, embedded, and IoT devices. It enables:

- **Low Latency**: Fast inference on resource-constrained devices
- **Small Binary Size**: Optimized for minimal app size
- **Low Power Consumption**: Efficient execution on battery-powered devices
- **Hardware Acceleration**: Support for GPU, DSP, and NPU acceleration

---

## 📊 Quantization: Making Models Deployment-Ready

### What is Quantization?

**Quantization** is the process of converting high-precision (32-bit floating-point) weights and activations to lower-precision formats (typically 8-bit integers). This technique is crucial for deploying models on edge devices.

### Benefits of Quantization

#### 1. **Model Compression**
- **4x smaller model size** with default quantization
- Reduced storage requirements on device
- Faster model download and updates
- Example: 8.86 MB model → 2.63 MB quantized model

#### 2. **Latency Reduction**
- **1.5-4x faster inference** on CPU backends
- Reduced memory bandwidth requirements
- Lower power consumption
- Better user experience with real-time applications

### Quantization-Aware Training (QAT)

**Quantization-aware training** emulates inference-time quantization during training, creating models that are optimized for quantization from the start.

#### How QAT Works:
1. **During Training**: Simulates quantization effects by inserting fake quantization nodes
2. **Forward Pass**: Weights and activations are quantized and dequantized
3. **Backward Pass**: Gradients flow through quantization operations
4. **Result**: Model learns to be robust to quantization noise

#### Benefits:
- **Higher Accuracy**: Better than post-training quantization alone
- **Predictable Performance**: Know accuracy before deployment
- **Optimal Weights**: Network adapts to low-precision during training

#### Deploy with Quantization:
Quantization brings improvements via model compression and latency reduction. With the API defaults, the model size shrinks by 4x, and we typically see between 1.5 - 4x improvements in CPU latency in the tested backends. Eventually, latency improvements can be seen on compatible machine learning accelerators, such as the EdgeTPU and NNAPI.

---

## 🔧 Three Quantization Approaches

### 1. **No Quantization (Baseline)**
\`\`\`python
converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
tflite_model = converter.convert()
\`\`\`
- **Size**: ~8.86 MB
- **Precision**: 32-bit float
- **Accuracy**: Highest (baseline)
- **Speed**: Slowest on edge devices
- **Use Case**: When accuracy is critical and device has sufficient resources

### 2. **Dynamic Range Quantization**
\`\`\`python
converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
\`\`\`
- **Size**: ~2.63 MB (70% reduction!)
- **Precision**: 8-bit int weights, float activations
- **Accuracy**: Minimal loss (typically <1%)
- **Speed**: 2-4x faster on CPU
- **Use Case**: Best balance for most applications

### 3. **Full Integer Quantization**
\`\`\`python
converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_data_gen():
    for data in dataset.take(100):
        yield [data]

converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
tflite_model = converter.convert()
\`\`\`
- **Size**: ~2.84 MB
- **Precision**: 8-bit int weights AND activations
- **Accuracy**: Slight loss (typically 1-2%)
- **Speed**: Fastest on dedicated hardware (EdgeTPU, NNAPI)
- **Use Case**: Maximum performance on specialized accelerators

---

## 📈 Performance Comparison

| Model Type | Size | CPU Latency | EdgeTPU Latency | Accuracy Loss |
|------------|------|-------------|-----------------|---------------|
| Baseline (Float32) | 8.86 MB | 1x (baseline) | Not supported | 0% |
| Dynamic Range | 2.63 MB | 2-3x faster | Not optimal | <1% |
| Full Integer | 2.84 MB | 1.5-2x faster | 10-15x faster | 1-2% |

*Note: Latency improvements depend on hardware and model architecture*


## 🎓 Learning Path

### Beginners
1. Start with \`train_and_convert_to_tflite.py\` to understand the basic workflow
2. Experiment with different model architectures
3. Compare float vs. quantized model performance

### Intermediate
1. Complete \`quantization_optimization_assignment.py\`
2. Create all three model variants (baseline, dynamic, full integer)
3. Analyze size-accuracy-speed tradeoffs

### Advanced
1. Implement quantization-aware training
2. Optimize for specific hardware targets
3. Fine-tune representative dataset for calibration
4. Profile models on real edge devices


### Selective Quantization
\`\`\`python
# Keep certain layers in float32 for accuracy
converter.target_spec.supported_types = [tf.float16]
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS  # Allow float fallback
]
\`\`\`

### Mixed-Precision Quantization
Combine different precision levels for optimal accuracy-performance tradeoff.
