#!/usr/bin/env python3

import sys
import os
import gi
import threading
import time
import cv2
import argparse
import numpy as np
import logging
import configparser
import signal
import queue
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('integrated_pipeline')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import DeepStream modules
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, GLib

# Import DeepStream Python bindings
try:
    import pyds
except ImportError as e:
    logger.error(f"Error importing DeepStream Python bindings: {e}")
    logger.error("Please check that DeepStream SDK is installed correctly.")
    sys.exit(1)

# Import local modules
try:
    from snap_detector_plugin import SnapDetector
    from model_adapter import ModelAdapter
except ImportError as e:
    logger.error(f"Error importing local modules: {e}")
    logger.error("Please check that all required modules are available.")
    sys.exit(1)

# Global state variables
class PipelineState:
    def __init__(self):
        self.pipeline = None  # GStreamer pipeline
        self.loop = None      # GLib main loop
        self.active = False   # Whether pipeline processing is active
        self.running = True   # Whether the application is still running
        self.snap_detected = False
        
        # Thread-safe data structures
        self.lock = threading.RLock()
        self.detection_results = {}
        self.location_info = {"city": "Unknown", "district": "Unknown"}
        self.weather_info = {"temperature": "N/A", "condition": "Unknown"}
        self.last_location_update = 0
        
        # Thread-safe detection queue
        self.detection_queue = queue.Queue()
        
        # Model adapter
        self.model_adapter = ModelAdapter()
        
        # Snap detector
        self.snap_detector = None

state = PipelineState()

def toggle_pipeline_state():
    """Toggle the pipeline active state"""
    with state.lock:
        state.active = not state.active
        status = "ACTIVE" if state.active else "INACTIVE"
        logger.info(f"Pipeline state toggled: {status}")

def snap_detector_thread(video_path: str) -> None:
    """
    Thread that monitors for finger snap gestures
    
    Args:
        video_path: Path to the input video
    """
    global state
    
    try:
        logger.info("Starting snap detector thread")
        
        # Initialize snap detector
        state.snap_detector = SnapDetector(alternative_key='s')
        
        # Register callback for snap detection
        state.snap_detector.add_callback(toggle_pipeline_state)
        
        # Open video for snap detection
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video for snap detection: {video_path}")
            return
        
        logger.info("Snap detector thread ready - listening for snap gestures")
        
        # Main processing loop
        while state.running:
            try:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind video
                    continue
                    
                # Detect snap gestures in the frame
                state.snap_detector.detect(frame)
                
                # Reduce CPU load
                time.sleep(0.03)
            except Exception as e:
                logger.error(f"Error in snap detection: {e}", exc_info=True)
                time.sleep(0.1)  # Add delay on error
    
    except Exception as e:
        logger.error(f"Fatal error in snap detector thread: {e}", exc_info=True)
    
    finally:
        logger.info("Snap detector thread exiting")
        if cap is not None and cap.isOpened():
            cap.release()
        if state.snap_detector is not None:
            state.snap_detector.close()

def location_weather_thread() -> None:
    """Thread that updates location and weather information periodically"""
    global state
    
    try:
        logger.info("Starting location and weather thread")
        
        # Load models
        state.model_adapter.load_model("location")
        state.model_adapter.load_model("weather")
        
        # Main processing loop
        while state.running:
            try:
                current_time = time.time()
                
                # Update location and weather every 60 seconds
                if current_time - state.last_location_update > 60:
                    # Get frame from detection queue if available
                    try:
                        if not state.detection_queue.empty():
                            frame = state.detection_queue.get(timeout=0.1)
                            
                            # Process location
                            location = state.model_adapter.process_frame("location", frame)
                            if location and location.get("city", "Unknown") != "Unknown":
                                with state.lock:
                                    state.location_info = location
                                
                                # Get weather for detected location
                                weather = state.model_adapter.process_frame("weather", None, city=location["city"])
                                if weather and weather.get("success", False):
                                    with state.lock:
                                        state.weather_info = weather
                                
                                state.last_location_update = current_time
                                logger.info(f"Updated location and weather: {location['city']}, {weather['temperature']}")
                            
                            state.detection_queue.task_done()
                    except queue.Empty:
                        pass
                    except Exception as e:
                        logger.error(f"Error processing location/weather: {e}", exc_info=True)
                
                # Reduce CPU load
                time.sleep(1)
            
            except Exception as e:
                logger.error(f"Error in location/weather thread: {e}", exc_info=True)
                time.sleep(1)  # Add delay on error
    
    except Exception as e:
        logger.error(f"Fatal error in location/weather thread: {e}", exc_info=True)
    
    finally:
        logger.info("Location and weather thread exiting")

def model_detection_thread(video_path: str) -> None:
    """
    Thread that runs computer vision models on video frames
    
    Args:
        video_path: Path to the input video
    """
    global state
    
    try:
        logger.info("Starting model detection thread")
        
        # Load all models
        state.model_adapter.load_model("yoloe")
        state.model_adapter.load_model("rf_detr")
        state.model_adapter.load_model("minimap")
        
        # Open video for model detection
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video for model detection: {video_path}")
            return
        
        frame_count = 0
        logger.info("Model detection thread ready")
        
        # Main processing loop
        while state.running:
            try:
                if state.active:
                    ret, frame = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind video
                        frame_count = 0
                        continue
                    
                    frame_count += 1
                    
                    # Add frame to location/weather queue (at lower frequency)
                    if frame_count % 30 == 0:  # Every 30 frames
                        try:
                            if state.detection_queue.full():
                                try:
                                    state.detection_queue.get_nowait()  # Remove old frame
                                except queue.Empty:
                                    pass
                            state.detection_queue.put_nowait(frame.copy())
                        except:
                            pass
                    
                    # Process models at different frequencies to manage performance
                    try:
                        # YOLOE detection (every frame)
                        state.model_adapter.process_frame_async("yoloe", frame.copy(), confidence=0.25)
                        
                        # RF-DETR human detection (every 3 frames)
                        if frame_count % 3 == 0:
                            state.model_adapter.process_frame_async("rf_detr", frame.copy())
                        
                        # Minimap (every 5 frames)
                        if frame_count % 5 == 0:
                            state.model_adapter.process_frame_async("minimap", frame.copy())
                    
                    except Exception as e:
                        logger.error(f"Error in model processing: {e}", exc_info=True)
                
                # Reduce CPU load
                time.sleep(0.01)
            
            except Exception as e:
                logger.error(f"Error in model detection thread: {e}", exc_info=True)
                time.sleep(0.1)  # Add delay on error
    
    except Exception as e:
        logger.error(f"Fatal error in model detection thread: {e}", exc_info=True)
    
    finally:
        logger.info("Model detection thread exiting")
        if cap is not None and cap.isOpened():
            cap.release()

def osd_sink_pad_buffer_probe(pad, info, u_data):
    """
    DeepStream probe callback for visualization
    
    This function is called by DeepStream for each frame in the pipeline.
    It visualizes the detection results and metadata on the frame.
    """
    global state
    
    try:
        # Get the GstBuffer
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK
        
        # Get batch metadata
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK
        
        # Iterate through frames in batch
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break
            
            # Create display metadata
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            if display_meta:
                # Set pipeline status text
                with state.lock:
                    if state.active:
                        status_text = "Pipeline: ACTIVE (Snap to deactivate)"
                        status_color = (0.0, 1.0, 0.0, 1.0)  # Green
                    else:
                        status_text = "Pipeline: INACTIVE (Snap to activate)"
                        status_color = (1.0, 0.0, 0.0, 1.0)  # Red
                
                # Configure status text params
                display_meta.num_labels = 1
                py_nvosd_text_params = display_meta.text_params[0]
                py_nvosd_text_params.display_text = status_text
                py_nvosd_text_params.x_offset = 10
                py_nvosd_text_params.y_offset = 30
                py_nvosd_text_params.font_params.font_name = "Arial"
                py_nvosd_text_params.font_params.font_size = 15
                py_nvosd_text_params.font_params.font_color.set(*status_color)
                py_nvosd_text_params.set_bg_clr = 1
                py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.5)
                
                # Show location and weather info if pipeline is active
                with state.lock:
                    if state.active:
                        # Get location and weather information
                        loc_text = f"Location: {state.location_info['city']}, {state.location_info.get('district', 'Unknown')}"
                        weather_text = f"Weather: {state.weather_info.get('temperature', 'N/A')}, {state.weather_info.get('condition', 'Unknown')}"
                        
                        # Configure additional text displays
                        display_meta.num_labels = 3
                        
                        # Location info
                        py_nvosd_text_params = display_meta.text_params[1]
                        py_nvosd_text_params.display_text = loc_text
                        py_nvosd_text_params.x_offset = 10
                        py_nvosd_text_params.y_offset = 70
                        py_nvosd_text_params.font_params.font_name = "Arial"
                        py_nvosd_text_params.font_params.font_size = 15
                        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
                        py_nvosd_text_params.set_bg_clr = 1
                        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.5)
                        
                        # Weather info
                        py_nvosd_text_params = display_meta.text_params[2]
                        py_nvosd_text_params.display_text = weather_text
                        py_nvosd_text_params.x_offset = 10
                        py_nvosd_text_params.y_offset = 110
                        py_nvosd_text_params.font_params.font_name = "Arial"
                        py_nvosd_text_params.font_params.font_size = 15
                        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
                        py_nvosd_text_params.set_bg_clr = 1
                        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.5)
                
                # Add display metadata to frame
                pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
                
                # Visualize detections if pipeline is active
                with state.lock:
                    if state.active:
                        # Get model results from model adapter
                        yoloe_result = state.model_adapter.get_latest_result("yoloe")
                        rf_detr_result = state.model_adapter.get_latest_result("rf_detr") 
                        minimap_result = state.model_adapter.get_latest_result("minimap")
                        
                        # Visualize YOLOE detections
                        if yoloe_result and yoloe_result.get('result'):
                            # Here we would add object boxes to display_meta
                            # This is model-specific and depends on YOLOE output format
                            # Example:
                            # for detection in yoloe_result.get('result'):
                            #    Add rect_params to display_meta
                            pass
                        
                        # Visualize RF-DETR human detections with triangles
                        if rf_detr_result and rf_detr_result.get('result'):
                            # This would add custom triangle visualizations for people
                            # Similar to the implementation in deepstream_pipeline.py
                            pass
                        
                        # Add minimap visualization
                        if minimap_result and minimap_result.get('result') is not None:
                            # This would add the minimap as an overlay
                            # Depends on minimap format and implementation
                            pass
            
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
                
        return Gst.PadProbeReturn.OK
    
    except Exception as e:
        logger.error(f"Error in buffer probe: {e}", exc_info=True)
        return Gst.PadProbeReturn.OK

def bus_call(bus, message, loop):
    """
    GStreamer bus message handler
    
    Args:
        bus: GStreamer bus
        message: GStreamer message
        loop: GLib main loop
    
    Returns:
        bool: Always True to continue receiving messages
    """
    t = message.type
    if t == Gst.MessageType.EOS:
        logger.info("End-of-stream")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        logger.warning(f"GStreamer warning: {err}: {debug}")
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        logger.error(f"GStreamer error: {err}: {debug}")
        loop.quit()
    return True

def signal_handler(sig, frame):
    """Signal handler for clean shutdown"""
    logger.info("Received interrupt signal, shutting down...")
    cleanup()
    sys.exit(0)

def cleanup():
    """Clean up resources before exit"""
    global state
    
    logger.info("Cleaning up resources...")
    
    # Stop threads
    state.running = False
    time.sleep(1)  # Give threads time to exit
    
    # Stop pipeline
    if state.pipeline is not None:
        state.pipeline.set_state(Gst.State.NULL)
        logger.info("Pipeline stopped")
    
    # Clean up model adapter
    if hasattr(state, 'model_adapter'):
        state.model_adapter.close()
        logger.info("Model adapter closed")
    
    # Clean up snap detector
    if state.snap_detector is not None:
        state.snap_detector.close()
        logger.info("Snap detector closed")
    
    logger.info("Cleanup complete")

def main():
    """Main function"""
    global state
    
    # Register signal handler for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Integrated DeepStream Pipeline')
    parser.add_argument('--video', required=True, help='Path to input video file')
    parser.add_argument('--output', default=None, help='Path to output video file')
    parser.add_argument('--config', default='deepstream_app_config.txt', help='Path to DeepStream configuration file')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Check if video file exists
    video_path = args.video
    if not os.path.exists(video_path):
        logger.error(f"Input video file {video_path} does not exist.")
        return -1
        
    # Create output filename if not specified
    if args.output is None:
        video_dir = os.path.dirname(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(video_dir, f"{video_name}_integrated_{timestamp}.mp4")
    
    # Check if configuration file exists
    config_path = args.config
    if not os.path.exists(config_path):
        logger.error(f"Configuration file {config_path} does not exist.")
        return -1
    
    # GStreamer initialization
    logger.info("Initializing GStreamer")
    GObject.threads_init()
    Gst.init(None)
    
    # Update config file with video path
    config = configparser.ConfigParser()
    config.read(config_path)
    if 'source0' in config:
        config.set('source0', 'uri', f'file://{os.path.abspath(video_path)}')
    
    # Create a temporary configuration file with updated paths
    temp_config_path = 'temp_config.txt'
    with open(temp_config_path, 'w') as configfile:
        config.write(configfile)
    
    # Create Pipeline
    logger.info("Creating DeepStream pipeline")
    pipeline_str = (
        f"filesrc location={video_path} ! qtdemux ! h264parse ! "
        "nvv4l2decoder ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
        "nvstreammux name=mux batch-size=1 width=1280 height=720 ! "
        "nvinfer name=primary-infer config-file-path=models/yoloe_config.txt ! "
        "nvtracker name=tracker ll-config-file=models/tracker_config.yml "
        "ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so ! "
        "nvdsosd name=osd ! "
        "nvvideoconvert ! "
        "nvv4l2h264enc ! "
        "h264parse ! "
        "qtmux ! "
        f"filesink location={args.output}"
    )
    
    try:
        pipeline = Gst.parse_launch(pipeline_str)
        state.pipeline = pipeline
        
        # Set up OSD probe for metadata visualization
        osdsinkpad = pipeline.get_by_name("osd").get_static_pad("sink")
        if not osdsinkpad:
            logger.error("Unable to get sink pad of nvosd")
            return -1
        
        osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)
        
        # Create detection queue with limited size to avoid memory issues
        state.detection_queue = queue.Queue(maxsize=2)
        
        # Start threads
        logger.info("Starting worker threads")
        
        # Snap detector thread
        snap_thread = threading.Thread(target=snap_detector_thread, args=(video_path,))
        snap_thread.daemon = True
        snap_thread.start()
        
        # Location/weather thread
        location_thread = threading.Thread(target=location_weather_thread)
        location_thread.daemon = True
        location_thread.start()
        
        # Model detection thread
        model_thread = threading.Thread(target=model_detection_thread, args=(video_path,))
        model_thread.daemon = True
        model_thread.start()
        
        # Start pipeline
        logger.info(f"Starting video processing: {video_path}")
        logger.info(f"Output will be saved to: {args.output}")
        logger.info("Use finger snap or press 's' key to activate/deactivate pipeline")
        
        # Create main loop
        state.loop = GLib.MainLoop()
        bus = state.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", bus_call, state.loop)
        
        # Start playing
        state.pipeline.set_state(Gst.State.PLAYING)
        
        try:
            state.loop.run()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        
        # Clean up
        cleanup()
        
        # Remove temp config
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
        
        logger.info(f"Pipeline completed. Output saved to: {args.output}")
        return 0
        
    except Exception as e:
        logger.error(f"Error creating pipeline: {e}", exc_info=True)
        # Clean up any resources
        cleanup()
        return -1

if __name__ == "__main__":
    sys.exit(main()) 