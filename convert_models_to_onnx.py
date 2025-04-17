#!/usr/bin/env python3

import os
import sys
import torch
import argparse
import logging
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_converter')

# Add the parent directory to path so we can import our models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def convert_yoloe_to_onnx(output_dir: str, dynamic: bool = True, fp16: bool = False, simplify: bool = True) -> bool:
    """
    Convert YOLOE model to ONNX format
    
    Args:
        output_dir: Directory to save ONNX model
        dynamic: Whether to use dynamic axes
        fp16: Whether to use half precision (FP16)
        simplify: Whether to simplify the ONNX model
        
    Returns:
        bool: Success status
    """
    logger.info("Converting YOLOE model to ONNX format...")
    
    try:
        # Try to import ultralytics
        try:
            from ultralytics import YOLO
        except ImportError as e:
            logger.error(f"Error importing ultralytics: {e}")
            logger.error("Please install ultralytics: pip install ultralytics")
            return False
        
        # Check for model file
        model_path = "../pipeline_model_files/models/yoloe-v8s-seg.pt"
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            logger.error("Please download the model or specify correct path")
            return False
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        onnx_path = os.path.join(output_dir, "yoloe-v8s-seg.onnx")
        
        # Load YOLOE model
        logger.info(f"Loading model from {model_path}")
        model = YOLO(model_path)
        
        # Export to ONNX
        logger.info(f"Exporting to ONNX: {onnx_path}")
        model.export(format="onnx", 
                    opset=13,
                    simplify=simplify, 
                    dynamic=dynamic,
                    half=fp16)
        
        # Move model if it was saved to a different location
        expected_onnx = os.path.splitext(model_path)[0] + ".onnx"
        if expected_onnx != onnx_path and os.path.exists(expected_onnx):
            shutil.move(expected_onnx, onnx_path)
            logger.info(f"Moved ONNX model to {onnx_path}")
        
        if os.path.exists(onnx_path):
            logger.info(f"Successfully exported YOLOE model to: {onnx_path}")
            return True
        else:
            logger.error(f"ONNX model not found at expected path: {onnx_path}")
            return False
    
    except Exception as e:
        logger.error(f"Error converting YOLOE model: {e}", exc_info=True)
        return False


def convert_rfdetr_to_onnx(output_dir: str, dynamic: bool = True, fp16: bool = False) -> bool:
    """
    Convert RF-DETR model to ONNX format
    
    Args:
        output_dir: Directory to save ONNX model
        dynamic: Whether to use dynamic axes
        fp16: Whether to use half precision (FP16)
        
    Returns:
        bool: Success status
    """
    logger.info("Converting RF-DETR model to ONNX format...")
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        onnx_path = os.path.join(output_dir, "rfdetr.onnx")
        
        # Try to import RF-DETR implementation
        try:
            from pipeline_model_files.rf_detr import RFDETRBase
        except ImportError as e:
            logger.error(f"Error importing RFDETRBase: {e}")
            logger.error("Please ensure the RF-DETR model is available")
            return False
        
        # Check CUDA availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        if device == "cpu" and fp16:
            logger.warning("FP16 precision requested but running on CPU. Disabling FP16.")
            fp16 = False
        
        # Load RF-DETR model
        logger.info("Loading RF-DETR model")
        model = RFDETRBase(device=device)
        
        # Convert model to fp16 if requested
        if fp16 and device == "cuda":
            logger.info("Converting model to FP16 precision")
            model = model.half()
        
        # Create dummy input (RF-DETR expects RGB input with shape [1, 3, H, W])
        logger.info("Creating dummy input tensor")
        input_height, input_width = 640, 640
        dummy_input = torch.randn(1, 3, input_height, input_width)
        
        # Move input to appropriate device and precision
        dummy_input = dummy_input.to(device)
        if fp16 and device == "cuda":
            dummy_input = dummy_input.half()
        
        # Define dynamic axes if requested
        dynamic_axes = None
        if dynamic:
            dynamic_axes = {
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size'}
            }
        
        # Export model to ONNX format
        logger.info(f"Exporting model to ONNX: {onnx_path}")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes
        )
        
        logger.info(f"Successfully exported RF-DETR model to: {onnx_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error converting RF-DETR model: {e}", exc_info=True)
        return False


def verify_onnx_model(onnx_path: str) -> bool:
    """
    Verify that the ONNX model is valid
    
    Args:
        onnx_path: Path to ONNX model file
        
    Returns:
        bool: Whether the model is valid
    """
    logger.info(f"Verifying ONNX model: {onnx_path}")
    
    try:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        logger.info(f"ONNX model {onnx_path} is valid")
        return True
    except ImportError as e:
        logger.error(f"Error importing onnx module: {e}")
        logger.error("Please install onnx: pip install onnx")
        return False
    except Exception as e:
        logger.error(f"Error verifying ONNX model {onnx_path}: {e}", exc_info=True)
        return False


def optimize_onnx_with_tensorrt(onnx_path: str, engine_path: str, 
                                precision: str = "fp32", 
                                workspace_size: int = 1) -> bool:
    """
    Optimize ONNX model with TensorRT
    
    Args:
        onnx_path: Path to ONNX model file
        engine_path: Path to save TensorRT engine
        precision: Precision to use (fp32, fp16, or int8)
        workspace_size: Workspace size in GB
        
    Returns:
        bool: Success status
    """
    logger.info(f"Optimizing ONNX model with TensorRT: {onnx_path}")
    
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        logger.info(f"TensorRT version: {trt.__version__}")
        
        # Create TensorRT logger
        trt_logger = trt.Logger(trt.Logger.INFO)
        
        # Create builder
        builder = trt.Builder(trt_logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        
        # Create parser and parse ONNX model
        parser = trt.OnnxParser(network, trt_logger)
        with open(onnx_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                for error in range(parser.num_errors):
                    logger.error(f"TensorRT ONNX parser error: {parser.get_error(error)}")
                return False
        
        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = workspace_size * 1 << 30  # Convert to bytes
        
        # Set precision
        if precision == "fp16" and builder.platform_has_fast_fp16:
            logger.info("Enabling FP16 precision")
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8" and builder.platform_has_fast_int8:
            logger.info("Enabling INT8 precision")
            config.set_flag(trt.BuilderFlag.INT8)
            # INT8 calibration would be needed here
        
        # Build engine
        logger.info("Building TensorRT engine (this may take a while)...")
        engine = builder.build_engine(network, config)
        
        if engine is None:
            logger.error("Failed to build TensorRT engine")
            return False
        
        # Save engine
        logger.info(f"Saving TensorRT engine to: {engine_path}")
        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())
        
        logger.info(f"Successfully optimized model with TensorRT: {engine_path}")
        return True
        
    except ImportError as e:
        logger.error(f"Error importing TensorRT modules: {e}")
        logger.error("Please install TensorRT: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/")
        return False
    except Exception as e:
        logger.error(f"Error optimizing with TensorRT: {e}", exc_info=True)
        return False


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Convert PyTorch models to ONNX format")
    parser.add_argument("--output-dir", type=str, default="../pipeline_model_files/models",
                        help="Directory to save ONNX models")
    parser.add_argument("--model", type=str, choices=["all", "yoloe", "rfdetr"], default="all",
                        help="Which model to convert")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision")
    parser.add_argument("--static", action="store_true", help="Use static input shapes (no dynamic axes)")
    parser.add_argument("--no-simplify", action="store_true", help="Disable ONNX simplification")
    parser.add_argument("--tensorrt", action="store_true", help="Optimize with TensorRT")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        default="INFO", help="Set logging level")
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")
    
    # Enable CUDA for PyTorch if available
    if torch.cuda.is_available():
        logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA is not available. Using CPU for conversion.")
        if args.fp16:
            logger.warning("FP16 precision requested but CUDA is not available. Disabling FP16.")
            args.fp16 = False
    
    # Convert models
    conversion_results = {}
    
    # YOLOE conversion
    if args.model in ["all", "yoloe"]:
        logger.info("Starting YOLOE conversion")
        yoloe_success = convert_yoloe_to_onnx(
            args.output_dir,
            dynamic=not args.static,
            fp16=args.fp16,
            simplify=not args.no_simplify
        )
        conversion_results["yoloe"] = yoloe_success
        
        # Verify YOLOE model
        if yoloe_success:
            yoloe_onnx_path = os.path.join(args.output_dir, "yoloe-v8s-seg.onnx")
            verify_onnx_model(yoloe_onnx_path)
            
            # Optimize with TensorRT if requested
            if args.tensorrt:
                yoloe_engine_path = os.path.join(args.output_dir, "yoloe-v8s-seg.engine")
                precision = "fp16" if args.fp16 else "fp32"
                optimize_onnx_with_tensorrt(yoloe_onnx_path, yoloe_engine_path, precision)
    
    # RF-DETR conversion
    if args.model in ["all", "rfdetr"]:
        logger.info("Starting RF-DETR conversion")
        rfdetr_success = convert_rfdetr_to_onnx(
            args.output_dir,
            dynamic=not args.static,
            fp16=args.fp16
        )
        conversion_results["rfdetr"] = rfdetr_success
        
        # Verify RF-DETR model
        if rfdetr_success:
            rfdetr_onnx_path = os.path.join(args.output_dir, "rfdetr.onnx")
            verify_onnx_model(rfdetr_onnx_path)
            
            # Optimize with TensorRT if requested
            if args.tensorrt:
                rfdetr_engine_path = os.path.join(args.output_dir, "rfdetr.engine")
                precision = "fp16" if args.fp16 else "fp32"
                optimize_onnx_with_tensorrt(rfdetr_onnx_path, rfdetr_engine_path, precision)
    
    # Print summary
    logger.info("Model conversion summary:")
    all_success = True
    for model, success in conversion_results.items():
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"  {model}: {status}")
        all_success = all_success and success
    
    if all_success:
        logger.info("All conversions completed successfully!")
        return 0
    else:
        logger.error("Some conversions failed. See log for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 