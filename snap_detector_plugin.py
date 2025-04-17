#!/usr/bin/env python3

import numpy as np
import cv2
import time
import math
import logging
import threading
import queue
from typing import Dict, Any, Optional, List, Tuple, Union, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('snap_detector')

# Try to import mediapipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    logger.info("Successfully imported mediapipe")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("mediapipe not installed. Snap detection will use keyboard instead.")
    
# Try to import keyboard module for alternative detection method
try:
    import keyboard
    KEYBOARD_AVAILABLE = True
    logger.info("Keyboard module available for alternative snap detection")
except ImportError:
    KEYBOARD_AVAILABLE = False
    logger.warning("keyboard module not installed. Alternative snap detection disabled.")

class SnapDetector:
    """
    Detector class for finger snap gestures. Uses either mediapipe for vision-based detection
    or keyboard input as a fallback.
    """
    def __init__(self, 
                alternative_key: str = 's',
                required_close_frames: int = 5,
                snap_cooldown: float = 3.0,
                distance_threshold: float = 0.03) -> None:
        """
        Initialize snap detector
        
        Args:
            alternative_key: Key to use as alternative to snap detection
            required_close_frames: Minimum consecutive frames with fingers close
            snap_cooldown: Cooldown period (seconds) between snap detections
            distance_threshold: Maximum distance between fingers to be considered close
        """
        # Configuration
        self.alternative_key = alternative_key
        self.required_close_frames = required_close_frames  
        self.snap_cooldown = snap_cooldown
        self.distance_threshold = distance_threshold
        
        # State variables
        self.snap_detected = False
        self.snap_confirmed = False
        self.close_counter = 0
        self.last_snap_time = 0
        self.snap_callbacks: List[Callable[[], None]] = []
        
        # Initialize mediapipe if available
        if MEDIAPIPE_AVAILABLE:
            try:
                self.mp_hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.6,
                    min_tracking_confidence=0.6
                )
                logger.info("Mediapipe hands model initialized")
            except Exception as e:
                logger.error(f"Error initializing mediapipe: {e}")
                self.mp_hands = None
        else:
            self.mp_hands = None
            
        # Setup keyboard detection if available
        if KEYBOARD_AVAILABLE:
            try:
                keyboard.on_press_key(self.alternative_key, self._on_key_event)
                logger.info(f"Keyboard detection initialized. Press '{self.alternative_key}' to simulate snap.")
            except Exception as e:
                logger.error(f"Error setting up keyboard listener: {e}")
            
        # Initialize worker thread for non-blocking operation
        self.frame_queue = queue.Queue(maxsize=1)  # Only keep most recent frame
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_frames, daemon=True)
        self.worker_thread.start()
        
        logger.info("SnapDetector initialized")
    
    def _on_key_event(self, event) -> None:
        """
        Handle keyboard press event as alternative to snap detection
        
        Args:
            event: Keyboard event
        """
        current_time = time.time()
        if current_time - self.last_snap_time > self.snap_cooldown:
            logger.info(f"Key '{self.alternative_key}' pressed - simulating snap")
            self.snap_detected = True
            self.snap_confirmed = True
            self.last_snap_time = current_time
            
            # Call any registered callbacks
            self._trigger_callbacks()
    
    def _process_frames(self) -> None:
        """Background thread to process frames from the queue"""
        while self.running:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get(timeout=0.1)
                    self._detect_from_frame(frame)
                    self.frame_queue.task_done()
                else:
                    time.sleep(0.01)  # Prevent busy waiting
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in frame processing: {e}")
                time.sleep(0.1)  # Add delay on error
                
    def _detect_from_frame(self, frame: np.ndarray) -> None:
        """
        Perform snap detection on a frame
        
        Args:
            frame: OpenCV frame to analyze
        """
        if not MEDIAPIPE_AVAILABLE or self.mp_hands is None:
            return
            
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    thumb_tip = hand_landmarks.landmark[4]  # Thumb tip
                    middle_tip = hand_landmarks.landmark[12]  # Middle finger tip
                    
                    dist = self.distance(thumb_tip, middle_tip)
                    
                    # If fingers are close
                    if dist < self.distance_threshold:
                        self.close_counter += 1
                    else:
                        self.close_counter = 0
                    
                    # If enough close frames and cooldown passed
                    if (self.close_counter >= self.required_close_frames and 
                        time.time() - self.last_snap_time > self.snap_cooldown):
                        
                        self.snap_detected = True
                        self.snap_confirmed = True
                        self.last_snap_time = time.time()
                        self.close_counter = 0
                        
                        logger.info("Snap gesture detected via mediapipe")
                        
                        # Call any registered callbacks
                        self._trigger_callbacks()
        except Exception as e:
            logger.error(f"Error analyzing frame for snap detection: {e}")
    
    def _trigger_callbacks(self) -> None:
        """Trigger all registered snap callbacks"""
        for callback in self.snap_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in snap callback: {e}")
    
    def distance(self, p1, p2) -> float:
        """
        Calculate Euclidean distance between two landmarks
        
        Args:
            p1: First landmark
            p2: Second landmark
            
        Returns:
            float: Distance between landmarks
        """
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    def detect(self, frame: np.ndarray) -> bool:
        """
        Process frame for snap detection
        
        Args:
            frame: OpenCV frame to analyze
            
        Returns:
            bool: Whether a snap was detected
        """
        # Reset snap detected status for next read
        detected = self.snap_detected
        self.snap_detected = False
        
        # Only queue the frame if mediapipe is available
        if MEDIAPIPE_AVAILABLE and self.mp_hands is not None and frame is not None:
            # Put frame in queue, dropping the oldest if queue is full
            try:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put_nowait(frame.copy())
            except:
                pass
        
        return detected
    
    def add_callback(self, callback: Callable[[], None]) -> None:
        """
        Register callback function to be called when snap is detected
        
        Args:
            callback: Function to call on snap detection
        """
        self.snap_callbacks.append(callback)
        logger.debug(f"Added snap callback. Total callbacks: {len(self.snap_callbacks)}")
    
    def visualize(self, frame: np.ndarray) -> np.ndarray:
        """
        Visualize snap detection status on frame
        
        Args:
            frame: OpenCV frame to visualize on
            
        Returns:
            np.ndarray: Frame with visualization
        """
        if self.snap_confirmed and time.time() - self.last_snap_time < 2:
            cv2.putText(frame, "Snap Detected!", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.5, (0, 255, 0), 4)
        
        # Add indicator for keyboard alternative
        if not MEDIAPIPE_AVAILABLE and KEYBOARD_AVAILABLE:
            cv2.putText(frame, f"Press '{self.alternative_key}' to snap", (50, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame
    
    def close(self) -> None:
        """Clean up resources"""
        logger.info("Shutting down SnapDetector")
        self.running = False
        
        if hasattr(self, 'worker_thread') and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
            
        if self.mp_hands:
            self.mp_hands.close()
            logger.debug("Closed mediapipe hands model")
            
        logger.info("SnapDetector shutdown complete")


def process_video(video_path: str, display: bool = True) -> List[float]:
    """
    Process video for snap detection
    
    Args:
        video_path: Path to video file
        display: Whether to display visualization
        
    Returns:
        list: Timestamps of detected snaps (seconds)
    """
    if not MEDIAPIPE_AVAILABLE and not KEYBOARD_AVAILABLE:
        logger.error("Both mediapipe and keyboard modules are unavailable. Cannot process video.")
        return []
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file {video_path}")
        return []
    
    # Initialize detector
    detector = SnapDetector()
    snap_times = []
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    
    logger.info(f"Processing video: {video_path}, {fps} FPS")
    
    # Process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        current_time = frame_count / fps
        
        # Detect snap
        if detector.detect(frame):
            snap_times.append(current_time)
            logger.info(f"Snap detected at {current_time:.2f} seconds")
        
        # Display visualization if enabled
        if display:
            vis_frame = detector.visualize(frame)
            cv2.imshow("Snap Detector", vis_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Clean up
    cap.release()
    if display:
        cv2.destroyAllWindows()
    
    detector.close()
    return snap_times


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect finger snaps in video")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--no-display", action="store_true", help="Disable real-time visualization")
    parser.add_argument("--key", type=str, default='s', help="Keyboard key to use as alternative to snap detection")
    
    args = parser.parse_args()
    
    # Process video
    snap_times = process_video(args.video, not args.no_display)
    
    # Report results
    if snap_times:
        logger.info(f"Detected {len(snap_times)} snaps at: {', '.join([f'{t:.2f}s' for t in snap_times])}")
    else:
        logger.info("No snaps detected in the video.") 