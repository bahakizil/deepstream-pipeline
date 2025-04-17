#!/usr/bin/env python3

import os
import sys
import cv2
import time
import logging
import numpy as np
from PIL import Image
import threading
from typing import Dict, Any, Optional, List, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
import queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_adapter')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check for GPU availability
try:
    import torch
    torch_available = torch.cuda.is_available()
    device = "cuda" if torch_available else "cpu"
    logger.info(f"PyTorch detected. Using device: {device}")
except ImportError:
    torch_available = False
    device = "cpu"
    logger.warning("PyTorch not available. Using CPU for inference.")

class ModelAdapter:
    """
    Adapter class for integrating various computer vision models with DeepStream pipeline.
    Each model has its dedicated adapter that makes the model's functions compatible with DeepStream.
    
    This class uses thread-safe data structures to enable concurrent model execution.
    """
    def __init__(self) -> None:
        """Initialize the model adapter with empty model dictionary and thread pool."""
        self.models: Dict[str, Dict[str, Any]] = {}
        self.model_results: Dict[str, Any] = {}
        self.model_queue = queue.Queue()
        self.result_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=3)  # Limit concurrent model execution
        self.running = True
        
        # Start the worker thread that processes the model queue
        self.worker_thread = threading.Thread(target=self._process_model_queue, daemon=True)
        self.worker_thread.start()
        
        logger.info("ModelAdapter initialized with thread pool")
        
    def _process_model_queue(self) -> None:
        """Process model requests from the queue in a separate thread."""
        while self.running:
            try:
                if not self.model_queue.empty():
                    model_name, frame, kwargs = self.model_queue.get(timeout=0.1)
                    try:
                        result = self._process_model_direct(model_name, frame, **kwargs)
                        with self.result_lock:
                            self.model_results[model_name] = {
                                'result': result,
                                'timestamp': time.time()
                            }
                    except Exception as e:
                        logger.error(f"Error processing model {model_name}: {e}")
                    finally:
                        self.model_queue.task_done()
                else:
                    time.sleep(0.01)  # Prevent busy waiting
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in model queue processing: {e}")
                time.sleep(0.1)  # Add delay on error
                
    def load_model(self, model_name: str) -> bool:
        """
        Load the specified model by name
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            bool: Whether loading was successful
        """
        if model_name in self.models and self.models[model_name].get("initialized", False):
            logger.debug(f"Model {model_name} already loaded")
            return True
            
        try:
            load_method = getattr(self, f"_load_{model_name}_model", None)
            if load_method is None:
                logger.error(f"Unknown model: {model_name}")
                return False
                
            logger.info(f"Loading model: {model_name}")
            success = load_method()
            if success:
                logger.info(f"Successfully loaded model: {model_name}")
            else:
                logger.error(f"Failed to load model: {model_name}")
            return success
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}", exc_info=True)
            return False
    
    def _load_yoloe_model(self) -> bool:
        """Load the YOLOE model."""
        try:
            from pipeline_model_files.yoloe_video_detector import YOLOEPromptDetector
            self.models["yoloe"] = {
                "model": YOLOEPromptDetector(),
                "initialized": True
            }
            logger.info("YOLOE model loaded successfully")
            return True
        except ImportError as e:
            logger.error(f"Cannot import YOLOEPromptDetector: {e}")
            self.models["yoloe"] = {"model": None, "initialized": False}
            return False
    
    def _load_rf_detr_model(self) -> bool:
        """Load the RF-DETR model."""
        try:
            from pipeline_model_files.rf_detr import RFDETRBase
            from pipeline_model_files.human_visualizer import visualize_humans
            
            # Load RF-DETR model
            rf_detr_model = RFDETRBase(device=device)
            
            self.models["rf_detr"] = {
                "model": rf_detr_model,
                "visualizer": visualize_humans,
                "initialized": True
            }
            logger.info("RF-DETR model loaded successfully")
            return True
        except ImportError as e:
            logger.error(f"Cannot import RF-DETR model: {e}")
            self.models["rf_detr"] = {"model": None, "initialized": False}
            return False
    
    def _load_minimap_model(self) -> bool:
        """Load the minimap processor."""
        try:
            # Load rf_detr first if not already loaded
            if "rf_detr" not in self.models:
                self._load_rf_detr_model()
                
            # Import rf_detr_minimap
            from pipeline_model_files.rf_detr_minimap import process_video_realtime
            
            self.models["minimap"] = {
                "processor": process_video_realtime,
                "initialized": True
            }
            logger.info("Minimap processor loaded successfully")
            return True
        except ImportError as e:
            logger.error(f"Cannot import Minimap processor: {e}")
            self.models["minimap"] = {"processor": None, "initialized": False}
            return False
    
    def _load_location_model(self) -> bool:
        """Load the location detector model."""
        try:
            from pipeline_model_files.location_detector import LocationDetector
            
            self.models["location"] = {
                "model": LocationDetector(),
                "initialized": True
            }
            logger.info("Location detector loaded successfully")
            return True
        except ImportError as e:
            logger.error(f"Cannot import LocationDetector: {e}")
            self.models["location"] = {"model": None, "initialized": False}
            return False
    
    def _load_weather_model(self) -> bool:
        """Load the weather detector model."""
        try:
            from pipeline_model_files.whether_detector import WeatherDetector
            
            self.models["weather"] = {
                "model": WeatherDetector(),
                "initialized": True
            }
            logger.info("Weather detector loaded successfully")
            return True
        except ImportError as e:
            logger.error(f"Cannot import WeatherDetector: {e}")
            self.models["weather"] = {"model": None, "initialized": False}
            return False
    
    def process_frame_async(self, model_name: str, frame: np.ndarray, **kwargs) -> None:
        """
        Queue a frame for asynchronous processing with the specified model
        
        Args:
            model_name: Name of the model to use for processing
            frame: OpenCV frame to process
            **kwargs: Model-specific parameters
        """
        if not frame.size:
            logger.warning(f"Received empty frame for model {model_name}")
            return
            
        self.model_queue.put((model_name, frame.copy(), kwargs))
    
    def get_latest_result(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest result for the specified model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Optional[Dict[str, Any]]: Latest result or None if not available
        """
        with self.result_lock:
            if model_name in self.model_results:
                return self.model_results[model_name]
        return None
        
    def process_frame(self, model_name: str, frame: np.ndarray, **kwargs) -> Any:
        """
        Process a frame with the specified model (synchronous)
        
        Args:
            model_name: Name of the model to use for processing
            frame: OpenCV frame to process
            **kwargs: Model-specific parameters
            
        Returns:
            Any: Model output
        """
        if not frame.size:
            logger.warning(f"Received empty frame for model {model_name}")
            return None
            
        if model_name not in self.models or not self.models[model_name].get("initialized", False):
            if not self.load_model(model_name):
                return None
        
        return self._process_model_direct(model_name, frame, **kwargs)
        
    def _process_model_direct(self, model_name: str, frame: np.ndarray, **kwargs) -> Any:
        """
        Direct model processing implementation
        
        Args:
            model_name: Name of the model to use
            frame: OpenCV frame to process
            **kwargs: Model-specific parameters
            
        Returns:
            Any: Model output
        """
        try:
            if model_name == "yoloe":
                return self._process_yoloe(frame, **kwargs)
            elif model_name == "rf_detr":
                return self._process_rf_detr(frame, **kwargs)
            elif model_name == "minimap":
                return self._process_minimap(frame, **kwargs)
            elif model_name == "location":
                return self._process_location(frame, **kwargs)
            elif model_name == "weather":
                city = kwargs.get("city", "Istanbul")
                return self._process_weather(city)
            else:
                logger.error(f"Unknown model {model_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing frame with model {model_name}: {e}", exc_info=True)
            return None
    
    def _process_yoloe(self, frame: np.ndarray, confidence: float = 0.25) -> Any:
        """Process a frame with YOLOE detector."""
        model = self.models["yoloe"]["model"]
        
        if hasattr(model, "process_single_frame"):
            return model.process_single_frame(frame, confidence=confidence)
        elif hasattr(model, "predict"):
            return model.predict(frame, conf=confidence)
        else:
            logger.warning("YOLOEPromptDetector does not have expected methods")
            return None
    
    def _process_rf_detr(self, frame: np.ndarray, class_id: Optional[int] = None) -> Any:
        """Process a frame with RF-DETR detector."""
        model = self.models["rf_detr"]["model"]
        
        if hasattr(model, "predict"):
            try:
                detections = model.predict(frame, threshold=0.5)
                
                # Filter by class ID if specified
                if class_id is not None and detections is not None:
                    mask = np.array([cls_id == class_id for cls_id in detections.class_id])
                    detections = detections[mask]
                
                return detections
            except Exception as e:
                logger.error(f"Error in RF-DETR prediction: {e}", exc_info=True)
                return None
        else:
            logger.warning("RF-DETR model does not have expected methods")
            return None
    
    def _process_minimap(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        """Process a frame to generate minimap visualization."""
        processor = self.models["minimap"]["processor"]
        
        if processor:
            try:
                return processor(frame, **kwargs)
            except Exception as e:
                logger.error(f"Error in minimap processing: {e}", exc_info=True)
                return frame
        else:
            logger.warning("Minimap processor not available")
            return frame
    
    def _process_location(self, frame: np.ndarray, **kwargs) -> Dict[str, str]:
        """Process a frame to detect location."""
        model = self.models["location"]["model"]
        
        # Convert OpenCV frame to PIL image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Analyze location
        if hasattr(model, "analyze_location"):
            try:
                location = model.analyze_location(pil_image, **kwargs)
                return location
            except Exception as e:
                logger.error(f"Error in location analysis: {e}", exc_info=True)
                return {"city": "Unknown", "district": "Unknown"}
        else:
            logger.warning("LocationDetector does not have expected methods")
            return {"city": "Unknown", "district": "Unknown"}
    
    def _process_weather(self, city: str) -> Dict[str, str]:
        """Get weather information for a city."""
        model = self.models["weather"]["model"]
        
        if hasattr(model, "get_weather"):
            try:
                weather = model.get_weather(city)
                return weather
            except Exception as e:
                logger.error(f"Error getting weather: {e}", exc_info=True)
                return {"temperature": "N/A", "condition": "Unknown", "success": False}
        else:
            logger.warning("WeatherDetector does not have expected methods")
            return {"temperature": "N/A", "condition": "Unknown", "success": False}
    
    def visualize_detections(self, model_name: str, frame: np.ndarray, 
                           detections: Any, **kwargs) -> np.ndarray:
        """
        Visualize model detections on a frame
        
        Args:
            model_name: Name of the model
            frame: Frame to visualize on
            detections: Model detection results
            **kwargs: Visualization parameters
            
        Returns:
            np.ndarray: Visualized frame
        """
        if frame is None or not frame.size:
            logger.warning(f"Empty frame passed to visualize_detections for {model_name}")
            return np.zeros((720, 1280, 3), dtype=np.uint8)
            
        if model_name == "yoloe":
            # YOLOE standard bounding box visualization
            if detections is not None:
                # Implement YOLOE visualization here
                return frame
            return frame
            
        elif model_name == "rf_detr":
            # RF-DETR human visualization
            if "rf_detr" in self.models and self.models["rf_detr"]["initialized"]:
                visualizer = self.models["rf_detr"]["visualizer"]
                if visualizer and detections is not None:
                    try:
                        return visualizer(frame, detections, **kwargs)
                    except Exception as e:
                        logger.error(f"Error in RF-DETR visualization: {e}", exc_info=True)
            return frame
            
        else:
            logger.warning(f"Visualization not implemented for model {model_name}")
            return frame
    
    def close(self) -> None:
        """Clean up resources and close all models."""
        logger.info("Shutting down ModelAdapter")
        self.running = False
        
        if hasattr(self, 'worker_thread') and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
            
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
            
        for model_name, model_info in self.models.items():
            if model_info.get("initialized", False) and hasattr(model_info.get("model"), "close"):
                try:
                    model_info["model"].close()
                    logger.info(f"Closed model: {model_name}")
                except Exception as e:
                    logger.error(f"Error closing model {model_name}: {e}")
        
        self.models = {}
        logger.info("ModelAdapter shutdown complete")


if __name__ == "__main__":
    # Parse test arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Test model adapter")
    parser.add_argument("--video", type=str, help="Path to test video")
    parser.add_argument("--model", type=str, choices=["yoloe", "rf_detr", "minimap", "location", "weather"], 
                       default="yoloe", help="Model to test")
    parser.add_argument("--async", action="store_true", help="Use asynchronous processing")
    
    args = parser.parse_args()
    
    # Test with video if specified
    if args.video and os.path.exists(args.video):
        adapter = ModelAdapter()
        
        # Load model
        if adapter.load_model(args.model):
            logger.info(f"Testing {args.model} model with {args.video}")
            
            cap = cv2.VideoCapture(args.video)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                if getattr(args, 'async', False):
                    adapter.process_frame_async(args.model, frame)
                    time.sleep(0.01)  # Give time for async processing
                    result = adapter.get_latest_result(args.model)
                    if result:
                        processed_result = result.get('result')
                    else:
                        processed_result = None
                else:
                    processed_result = adapter.process_frame(args.model, frame)
                
                # Visualize result
                vis_frame = adapter.visualize_detections(args.model, frame, processed_result)
                
                # Show result
                cv2.imshow("Model Test", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            
            adapter.close()
        else:
            logger.error(f"Failed to load model {args.model}")
    else:
        print("Please provide a valid video path with --video argument") 