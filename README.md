# Enhanced Integrated DeepStream Computer Vision Pipeline

This project creates a highly modular and optimized computer vision pipeline that utilizes NVIDIA DeepStream for real-time video processing on NVIDIA GPUs, with specific optimizations for the RTX 4070. The pipeline integrates multiple computer vision models into a single coherent system with event-driven control via finger-snap gestures.

## Key Features

- **Event-Driven Control**: Activate/deactivate the pipeline via finger-snap gestures or keyboard shortcuts
- **Multi-Model Integration**: Run YOLOE, RF-DETR, and other models concurrently on the same video stream
- **TensorRT Optimization**: Automatic ONNX model generation and optional TensorRT optimization
- **Thread-Safe Architecture**: Robust thread management with proper synchronization and error handling
- **GPU-Accelerated Processing**: Fully leverages NVIDIA DeepStream and TensorRT for GPU acceleration
- **Real-time Visualization**: Customized overlays for each detection model with support for advanced visualization
- **Modular Design**: Easily extend with new models or replace existing components

## System Requirements

- **Hardware**:
  - NVIDIA RTX 4070 GPU (or compatible NVIDIA GPU)
  - 16GB+ RAM recommended
  - SSD storage recommended for model storage and video processing

- **Software**:
  - Ubuntu 20.04 LTS or newer (preferred) or compatible Linux distribution
  - NVIDIA Drivers (>=525.105.17)
  - CUDA Toolkit (>=11.8)
  - TensorRT (>=8.6)
  - DeepStream SDK (>=7.1)
  - Python 3.8+

## Installation

### Option 1: Using Docker (Recommended)

This repository includes a Docker setup optimized for DeepStream 7.1:

```bash
# Build the Docker image
./runpod_setup.sh

# Run the container with a specific video
docker run --runtime=nvidia --gpus all -it deepstream-7.1-pipeline:latest --video /path/to/video.mp4 --output /path/to/output.mp4
```

### Option 2: Manual Installation

#### 1. Install NVIDIA Components

First, ensure you have installed the NVIDIA driver, CUDA, and TensorRT:

```bash
# Check NVIDIA driver installation
nvidia-smi

# Check CUDA installation
nvcc --version

# Check TensorRT installation (if already installed)
dpkg -l | grep tensorrt
```

#### 2. Install DeepStream SDK 7.1

Follow NVIDIA's official documentation to install DeepStream SDK:

```bash
# Download DeepStream SDK 7.1 from NVIDIA website
# https://developer.nvidia.com/deepstream-sdk-download

# For Ubuntu 20.04 with DeepStream 7.1:
sudo apt-get install ./deepstream-7.1_7.1.0-1_amd64.deb

# Install dependencies
sudo apt-get install libssl1.1 libjansson4 libgstreamer-plugins-base1.0-dev

# Configure environment
export CUDA_VER=11.8  # Adjust based on your CUDA version
```

#### 3. Clone this Repository

```bash
git clone <repository_url>
cd deepstream_pipeline
```

#### 4. Install Python Dependencies

Use the provided setup script to install all required dependencies:

```bash
# Option 1: Use the run_pipeline.sh script with install flag
./run_pipeline.sh --install

# Option 2: Install manually
pip install -r requirements.txt

# Install DeepStream Python bindings
cd /opt/nvidia/deepstream/deepstream/sources/python/
sudo python3 setup.py install
```

#### 5. Convert Models (Optional)

Convert PyTorch models to ONNX format for use with DeepStream:

```bash
# Option 1: Use the run_pipeline.sh script
./run_pipeline.sh --convert-models

# Option 2: Run conversion script directly
python3 convert_models_to_onnx.py
```

## RunPod Deployment

This pipeline is optimized for deployment on RunPod:

1. Build the Docker image:
   ```bash
   ./runpod_setup.sh
   ```

2. Upload the generated Docker image tar file to your preferred storage.

3. Create a new RunPod instance with:
   - **GPU**: NVIDIA RTX 4000 series or later (recommended)
   - **Container**: Custom Docker image (using your uploaded image)
   - **Environment Variables**:
     - `NVIDIA_VISIBLE_DEVICES=all`
     - `NVIDIA_DRIVER_CAPABILITIES=all`
   - **Volume Mounts**:
     - Host: `/path/for/videos` → Container: `/videos`
     - Host: `/path/for/output` → Container: `/output`
   - **Command**: 
     ```bash
     --video /videos/input.mp4 --output /output/result.mp4
     ```

For more detailed instructions, refer to the RunPod documentation or run `./runpod_setup.sh` for guidance.

## Usage

### Running the Pipeline

Use the provided run_pipeline.sh script for the simplest operation:

```bash
# Basic usage with required video input
./run_pipeline.sh --video path/to/your/video.mp4

# Full usage with all options
./run_pipeline.sh --video path/to/video.mp4 --output path/to/output.mp4 --config custom_config.txt --install --convert-models --tensorrt --fp16 --log-level INFO
```

### Command Line Arguments

The run_pipeline.sh script supports the following options:

| Option | Description |
|--------|-------------|
| -h, --help | Show help message |
| -v, --video PATH | Path to input video file (required) |
| -o, --output PATH | Path to output video file (optional) |
| -c, --config PATH | Path to DeepStream config file (default: deepstream_app_config.txt) |
| -i, --install | Install required dependencies |
| -m, --convert-models | Convert PyTorch models to ONNX format |
| -t, --tensorrt | Optimize ONNX models with TensorRT |
| --fp16 | Use FP16 precision for model conversion/optimization |
| -l, --log-level LEVEL | Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |

### Direct Python Execution

You can also run the pipeline directly with Python:

```bash
python3 integrated_pipeline.py --video path/to/video.mp4 --output path/to/output.mp4 --config deepstream_app_config.txt --log-level INFO
```

### Controlling the Pipeline

- **Finger Snap**: If mediapipe is installed, snap your fingers in view of the camera to toggle the pipeline state
- **Keyboard**: Press 's' key to toggle the pipeline state (alternative to finger snap)
- **Terminal Interrupt**: Press Ctrl+C in the terminal to gracefully stop the pipeline

## Project Structure

```
deepstream_pipeline/
├── integrated_pipeline.py     # Main pipeline controller
├── model_adapter.py           # Adapter class for model integration
├── snap_detector_plugin.py    # Finger snap detection module
├── deepstream_app_config.txt  # DeepStream configuration
├── convert_models_to_onnx.py  # Model conversion utility
├── run_pipeline.sh            # Convenience script for pipeline execution
├── requirements.txt           # Python dependencies
├── models/                    # Directory for model files
│   ├── yoloe_config.txt       # YOLOE model configuration
│   ├── rfdetr_config.txt      # RF-DETR model configuration
│   └── tracker_config.yml     # Tracker configuration
└── README.md                  # This documentation
```

## Architecture Overview

### Pipeline Components

1. **integrated_pipeline.py**: The main controller that integrates all components
   - Manages the DeepStream GStreamer pipeline
   - Coordinates all processing threads
   - Handles pipeline state and cleanup

2. **model_adapter.py**: Thread-safe model management
   - Dynamically loads models as needed
   - Provides unified interface for different model types
   - Supports both synchronous and asynchronous processing

3. **snap_detector_plugin.py**: Event detection module
   - Detects finger snap gestures using mediapipe
   - Provides keyboard fallback via the 's' key
   - Contains visualization tools for detected gestures

4. **DeepStream Integration**:
   - Uses GStreamer pipeline for video processing
   - Leverages DeepStream's hardware-accelerated elements (nvdec, nvinfer, etc.)
   - Uses nvdsosd for GPU-accelerated overlay rendering

### Thread Architecture

The pipeline uses multiple threads to maximize performance:

1. **Main Thread**: Runs the GStreamer pipeline and DeepStream processing
2. **Snap Detector Thread**: Continuously monitors for finger snap events
3. **Model Detection Thread**: Processes frames through vision models when active
4. **Location/Weather Thread**: Updates location and weather information periodically

All threads communicate through thread-safe queues and locks to prevent race conditions.

## Modular Detection Components

### YOLOE Detector

- General object detection for 80+ classes
- Supports segmentation and classification
- Optimized for real-time performance

### RF-DETR Human Detection

- Specialized human detection with higher accuracy
- Custom triangle visualization above detected people
- Supports tracking for consistent IDs

### Minimap Generation

- Creates a radar-style minimap visualization
- Useful for vehicle detection and spatial awareness
- Updates at a lower frequency to balance performance

### Location and Weather Module

- Analyzes images to determine geographic location
- Fetches weather information for detected locations
- Updates periodically (every 60 seconds)

## Performance Tuning

### GPU Optimization

The pipeline is specifically optimized for RTX 4070 GPUs:

- **Batched Processing**: Configured for efficient batch size
- **TensorRT Acceleration**: Optional model optimization with TensorRT
- **FP16 Precision**: Optional half-precision for faster inference
- **Memory Management**: Efficient CUDA memory handling

### Processing Strategies

Several strategies are employed to maintain real-time performance:

- **Staggered Processing**: Different models run at different frame intervals
- **Asynchronous Model Execution**: Non-blocking model inference
- **Resource Sharing**: Efficient frame data sharing between components
- **Dynamic Activation**: Process only when pipeline is active

## Extending the Pipeline

### Adding New Models

To add a new model:

1. Create adapter methods in `model_adapter.py`:
   ```python
   def _load_new_model(self):
       # Implementation
   
   def _process_new_model(self, frame, **kwargs):
       # Implementation
   ```

2. Add model loading to the model detection thread in `integrated_pipeline.py`

3. Add visualization code to the `osd_sink_pad_buffer_probe` function

### Customizing Visualization

DeepStream visualization is handled in the buffer probe callback:

- For rectangles: Use `display_meta.rect_params`
- For text: Use `display_meta.text_params`
- For lines/shapes: Use `display_meta.line_params`

## Troubleshooting

### Common Issues

1. **Missing CUDA/TensorRT**:
   - Error: "Failed to initialize CUDA context"
   - Solution: Verify CUDA and TensorRT installation with `nvidia-smi` and `nvcc --version`

2. **DeepStream Errors**:
   - Error: "Unable to get sink pad of nvosd"
   - Solution: Check DeepStream installation and environment variables

3. **Model Conversion Failures**:
   - Error: "Error converting model to ONNX"
   - Solution: Verify PyTorch installation and model paths

4. **Mediapipe Not Available**:
   - Warning: "mediapipe not installed. Using keyboard instead."
   - Solution: Install mediapipe or use keyboard 's' key for snap simulation

### Debugging

For detailed debugging information:

```bash
# Run with debug logging
./run_pipeline.sh --video input.mp4 --log-level DEBUG

# Check system GPU status
watch -n 1 nvidia-smi
```

## Docker Support

A Docker container is provided for easy deployment:

```bash
# Build the container
docker build -t deepstream-pipeline .

# Run the container with GPU support
docker run --gpus all -it --rm \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  -v /path/to/videos:/videos \
  deepstream-pipeline \
  --video /videos/input.mp4 \
  --output /videos/output.mp4
```

## License

This project is released under the MIT License.

## Acknowledgments

- NVIDIA for DeepStream SDK and TensorRT
- Google for Mediapipe framework
- Ultralytics for YOLOE models
- All contributors and open-source projects used in this pipeline

## Contact

For questions, issues, or contributions, please open an issue on the GitHub repository or contact the maintainers directly. 