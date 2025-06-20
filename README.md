# Low-Latency Conversational AI with Coqui XTTS v2

This repository contains a voice agent system implementing ultra-low-latency text-to-speech synthesis with Coqui's XTTS v2 multilingual model. The system is designed to achieve verbal response latency under 400ms for natural, real-time conversation.

## Architecture

The system has been optimized to use:

1. **Streaming TTS Generation**: Audio is generated and played in small chunks rather than waiting for the full synthesis.
2. **Cached Speaker Embeddings**: Speaker voice characteristics are precomputed during initialization.
3. **HiFi-GAN Vocoder**: Fast waveform synthesis for reducing inference time.
4. **CUDA Acceleration**: GPU-optimized inference using NVIDIA CUDA.
5. **KV Caching**: Optimization for transformer models to avoid recomputing attention weights.
6. **DeepSpeed Integration**: When using CUDA, DeepSpeed provides further optimization.
7. **Asynchronous Pipeline**: Producer-consumer architecture for parallel text processing and audio generation.
8. **GPU Memory Pinning**: Keeps model and embeddings permanently in VRAM for minimum latency.
9. **Mixed Precision**: Uses FP16 where available to speed up computation.
10. **Sentence Segmentation**: Process text in natural segments for improved parallelism.

## Hardware Requirements

To consistently achieve sub-400ms latency, we recommend:

### Minimum Hardware:
- **GPU**: NVIDIA RTX 3060 or equivalent with 8GB+ VRAM
- **CPU**: 8-core processor (Intel i7/AMD Ryzen 7 or better)
- **RAM**: 16GB minimum
- **Storage**: SSD for model loading (NVMe preferred)

### Recommended Hardware:
- **GPU**: NVIDIA RTX 3080/3090 or A100 with 16GB+ VRAM
- **CPU**: 16-core processor (Intel i9/AMD Ryzen 9 or better)
- **RAM**: 32GB or more
- **Storage**: NVMe SSD with 3000MB/s+ read speeds

### Software Configuration:
- Ubuntu 20.04 or newer / Windows 11 with WSL2
- CUDA 11.8 or newer
- cuDNN 8.6 or newer
- PyTorch 2.0+ with CUDA support
- Python 3.8 or newer

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Configure a reference audio file for voice cloning:
   - Record a clear voice sample (3-5 seconds) and save as WAV
   - Update the path in `voice_agent.py`: `self.default_speaker_wav = "/path/to/your/voice.wav"`

3. Run the voice agent:
   ```
   python voice_agent.py
   ```

4. For benchmarking latency:
   ```
   python voice_agent.py --benchmark
   ```

## Performance Optimization

### Memory Optimization Techniques

1. **Model Pinning to GPU Memory**
   - The model is loaded once and kept permanently in VRAM
   - Uses `pin_memory()` for tensors to avoid CPU-GPU transfers
   - Avoids any `model.to('cpu')` operations during runtime

2. **CUDA Memory Management**
   - Strategic use of `torch.cuda.empty_cache()` to prevent fragmentation
   - Pre-computation of speaker embeddings during initialization
   - CUDA stream synchronization to prevent memory conflicts

3. **Batch Size Optimization**
   - All inference runs with batch size 1 for lowest latency
   - Minimizes CPU-GPU transfer overhead and memory allocation

### Asynchronous Pipeline Architecture

The system uses a producer-consumer pattern for maximum throughput:

1. Text preprocessing and sentence segmentation occur in background threads
2. Audio generation happens asynchronously while previous chunks are playing
3. A fixed-size audio buffer balances generation and playback speeds

### Parameter Tuning

These parameters can be adjusted in `voice_agent.py` to optimize for your hardware:

1. **Audio Chunk Size** (default: 512)
   - Smaller values reduce time to first audio but may cause buffer underruns
   - Larger values improve audio smoothness but increase initial latency
   - Recommended range: 256-1024 based on hardware capabilities

2. **Reduction Factor** (default: 2)
   - Controls model speed vs. quality tradeoff
   - Lower values (1-2) provide better audio quality at higher latency
   - Higher values (3-8) provide faster inference at lower quality
   - Critical parameter for meeting 400ms target on lower-end hardware

3. **Max Decoder Steps** (default: 50)
   - Limits decoder iterations for speed
   - Lower values speed up generation but may truncate audio
   - Higher values improve quality but increase latency

4. **Maximum Sentence Length** (default: 150)
   - Controls text segmentation for parallel processing
   - Shorter segments improve responsiveness but may affect prosody
   - Longer segments sound more natural but take more time

## Monitoring and Optimization

The system includes a benchmark tool to measure performance across different:
- Text lengths
- Hardware configurations
- Parameter settings

Run the benchmark mode to find optimal settings:

```
python voice_agent.py --benchmark
```

## Troubleshooting

If not meeting the 400ms latency target:

1. **GPU Memory Issues**
   - Check GPU memory usage with `nvidia-smi`
   - Close other GPU applications
   - Set environment variable: `CUDA_VISIBLE_DEVICES=0` to isolate the GPU

2. **Inference Speed**
   - Increase `REDUCTION_FACTOR` (2-4 usually works well)
   - Decrease `MAX_SENTENCE_LENGTH` to enable more parallelism
   - Enable mixed precision: `export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`

3. **System Bottlenecks**
   - Ensure CPU isn't throttling (`cpufreq-info` on Linux)
   - Check thermal throttling (`nvidia-smi dmon -i 0 -s pucvt -d 1`)
   - Move models to faster storage (NVMe SSD)
   - Set process priority: `sudo nice -n -20 python voice_agent.py`

## Supported Languages

The system supports 17 languages including English, Spanish, French, German, Chinese, Japanese, and more.

## License

This project uses Coqui's XTTS v2, which is available under the Coqui Public Model License. 