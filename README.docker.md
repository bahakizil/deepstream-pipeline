# DeepStream Pipeline Docker Setup

This document describes how to build and use Docker images for the DeepStream pipeline.

## Development vs Production

This repository provides two Docker approaches:

1. **Development Container**: A lightweight container with basic dependencies for development and testing without requiring NVIDIA GPUs or the full DeepStream SDK.

2. **Production Container**: Instructions for building a full DeepStream container with GPU support for production deployments.

## Quick Start

```bash
# Build the development container
docker build -t deepstream-pipeline-dev:latest .

# Run the development container interactively
docker run --rm -it -v "$(pwd):/app" deepstream-pipeline-dev:latest
```

## Development Environment

The development environment is a lightweight Docker image that contains basic dependencies needed for development and testing without the full DeepStream SDK.

### Building the Development Image

```bash
# Build the development image
docker build -t deepstream-pipeline-dev:latest .
```

### Running the Development Container

```bash
# Run the development container with the current directory mounted
docker run --rm -it \
  -v $(pwd):/app \
  -v /path/to/videos:/videos \
  deepstream-pipeline-dev:latest \
  --input /videos/input.mp4 \
  --output /videos/output.mp4
```

## Development Container Usage

The development container is now ready to use. This provides a basic environment for working with the DeepStream pipeline code, but without the actual DeepStream SDK and NVIDIA GPU dependencies.

### Run the Container Interactively

```bash
# Run the container with the current directory mounted
docker run --rm -it -v "$(pwd):/app" deepstream-pipeline-dev:latest
```

This will give you an interactive bash shell inside the container where you can:
- Explore the codebase
- Run basic Python scripts
- Test code that doesn't depend on DeepStream or NVIDIA GPU

### For Production Deployment

When you're ready for production deployment with full DeepStream SDK:

1. Obtain access to an NVIDIA GPU machine with appropriate drivers
2. Install the DeepStream SDK using one of the methods described earlier in this document
3. Run your pipeline with GPU acceleration

## Production Environment with DeepStream SDK

For production deployments, you need the full DeepStream SDK with GPU support.

### Method 1: Using the NVIDIA NGC Docker Images

Pull the official DeepStream Docker image from NVIDIA NGC:

```bash
# Pull the DeepStream 7.1 container
docker pull nvcr.io/nvidia/deepstream:7.1
```

Run the container with GPU support:

```bash
# Run the DeepStream container with GPU support
docker run --gpus all -it --rm \
  -v $(pwd):/app \
  -v /path/to/videos:/videos \
  nvcr.io/nvidia/deepstream:7.1
```

### Method 2: Building a Custom DeepStream Container

To build a custom DeepStream container using the DeepStream SDK:

1. Download the DeepStream SDK from NVIDIA:
   - Visit: https://catalog.ngc.nvidia.com/orgs/nvidia/resources/deepstream
   - Download either:
     - Debian package: `deepstream-7.1_7.1.0-1_amd64.deb`
     - Tar package: `deepstream_sdk_v7.1.0_x86_64.tbz2`

2. Build the Docker image:

```dockerfile
FROM ubuntu:22.04

# Install DeepStream prerequisites
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 \
    libssl-dev \
    libgstreamer1.0-0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libgstreamer-plugins-base1.0-dev \
    libgstrtspserver-1.0-0 \
    libjansson4 \
    libyaml-cpp-dev \
    wget \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install DeepStream SDK
WORKDIR /tmp
COPY deepstream-7.1_7.1.0-1_amd64.deb .
RUN apt-get update && \
    apt-get install -y ./deepstream-7.1_7.1.0-1_amd64.deb && \
    rm deepstream-7.1_7.1.0-1_amd64.deb && \
    rm -rf /var/lib/apt/lists/*

# Set up DeepStream environment variables
ENV DEEPSTREAM_DIR=/opt/nvidia/deepstream/deepstream-7.1
ENV PYTHONPATH=${PYTHONPATH}:${DEEPSTREAM_DIR}/lib

# Copy application files
WORKDIR /app
COPY . /app

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Entry point
ENTRYPOINT ["/app/run_pipeline.sh"]
CMD ["--help"]
```

## DeepStream 7.1 Information

DeepStream 7.1 release information:
- Released: 2025
- Compatible with: Ubuntu 22.04
- CUDA: 12.6
- TensorRT: 10.3.0.26
- cuDNN: 9.3.0
- Supported GPUs: T4, A2, A10, A30, A100, RTX Ampere, Hopper, ADA 