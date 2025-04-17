#!/usr/bin/env python3

import sys
import os
import gi
import configparser
import argparse
from datetime import datetime

# Add parent directory to Python path for custom modules
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, GLib

# Import DeepStream Python bindings
try:
    import pyds
except ImportError as e:
    print(f"Error importing DeepStream Python bindings: {e}")
    print("Please check that DeepStream SDK is installed correctly.")
    sys.exit(1)

# Global variables
PIPELINE = None
LOOP = None

# Probe callback function for metadata visualization
def osd_sink_pad_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer")
        return Gst.PadProbeReturn.OK
    
    # Retrieve batch metadata from the gst_buffer
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK
    
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        
        frame_number = frame_meta.frame_num
        l_obj = frame_meta.obj_meta_list
        
        # Weather and location information from custom metadata
        # This would need to be implemented as a separate module in a real setup
        weather_text = "Weather: 25°C | Sunny"
        location_text = "Location: Istanbul, Beşiktaş"
        
        # Display weather and location info
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        if display_meta:
            display_meta.num_labels = 2
            
            # Display location info
            py_nvosd_text_params = display_meta.text_params[0]
            py_nvosd_text_params.display_text = location_text
            py_nvosd_text_params.x_offset = 10
            py_nvosd_text_params.y_offset = 30
            py_nvosd_text_params.font_params.font_name = "Arial"
            py_nvosd_text_params.font_params.font_size = 14
            py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
            py_nvosd_text_params.set_bg_clr = 1
            py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.5)
            
            # Display weather info
            py_nvosd_text_params = display_meta.text_params[1]
            py_nvosd_text_params.display_text = weather_text
            py_nvosd_text_params.x_offset = 10
            py_nvosd_text_params.y_offset = 60
            py_nvosd_text_params.font_params.font_name = "Arial"
            py_nvosd_text_params.font_params.font_size = 14
            py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
            py_nvosd_text_params.set_bg_clr = 1
            py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.5)
            
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        
        # Process object metadata and display on screen
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
                
            # Enhanced visualization for people (using triangle markers)
            if obj_meta.class_id == 0 and obj_meta.obj_label == "person":  # Person class
                # Add custom triangle visualization (similar to human_visualizer.py)
                triangle_display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                if triangle_display_meta:
                    triangle_display_meta.num_lines = 3
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = obj_meta.rect_params.left, obj_meta.rect_params.top, \
                                     obj_meta.rect_params.left + obj_meta.rect_params.width, \
                                     obj_meta.rect_params.top + obj_meta.rect_params.height
                    
                    # Draw triangle above the person
                    triangle_height = 20
                    triangle_base = 15
                    
                    # Triangle vertices
                    triangle_top_x = int((x1 + x2) / 2)
                    triangle_top_y = int(y1 - 5)
                    triangle_left_x = int(triangle_top_x - triangle_base / 2)
                    triangle_left_y = int(triangle_top_y - triangle_height)
                    triangle_right_x = int(triangle_top_x + triangle_base / 2)
                    triangle_right_y = triangle_left_y
                    
                    # Define triangle lines
                    line_params = triangle_display_meta.line_params
                    
                    # Line 1: top left to top right
                    line_params[0].x1 = triangle_left_x
                    line_params[0].y1 = triangle_left_y
                    line_params[0].x2 = triangle_right_x
                    line_params[0].y2 = triangle_right_y
                    line_params[0].line_width = 2
                    line_params[0].line_color.set(1.0, 0.0, 0.0, 1.0)  # Red
                    
                    # Line 2: top right to bottom
                    line_params[1].x1 = triangle_right_x
                    line_params[1].y1 = triangle_right_y
                    line_params[1].x2 = triangle_top_x
                    line_params[1].y2 = triangle_top_y
                    line_params[1].line_width = 2
                    line_params[1].line_color.set(1.0, 0.0, 0.0, 1.0)  # Red
                    
                    # Line 3: bottom to top left
                    line_params[2].x1 = triangle_top_x
                    line_params[2].y1 = triangle_top_y
                    line_params[2].x2 = triangle_left_x
                    line_params[2].y2 = triangle_left_y
                    line_params[2].line_width = 2
                    line_params[2].line_color.set(1.0, 0.0, 0.0, 1.0)  # Red
                    
                    pyds.nvds_add_display_meta_to_frame(frame_meta, triangle_display_meta)
            
            try:
                l_obj = l_obj.next
            except StopIteration:
                break
                
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
            
    return Gst.PadProbeReturn.OK


def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        print("End-of-stream\n")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        print("Warning: %s: %s\n" % (err, debug))
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print("Error: %s: %s\n" % (err, debug))
        loop.quit()
    return True


def main(args):
    global PIPELINE, LOOP
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='DeepStream Pipeline for Computer Vision Models')
    parser.add_argument('--video', required=True, help='Path to input video file')
    parser.add_argument('--output', default=None, help='Path to output video file')
    parser.add_argument('--config', default='deepstream_app_config.txt', help='Path to DeepStream configuration file')
    
    args = parser.parse_args(args)
    
    video_path = args.video
    if not os.path.exists(video_path):
        print(f"Error: Input video file {video_path} does not exist.")
        return -1
        
    # Create output filename if not specified
    if args.output is None:
        video_dir = os.path.dirname(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(video_dir, f"{video_name}_deepstream_{timestamp}.mp4")
    
    # Check if configuration file exists
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"Error: Configuration file {config_path} does not exist.")
        return -1
    
    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)
    
    # Create gstreamer pipeline from configuration file
    print("Creating Pipeline")
    
    # Read configuration file and update with input/output video paths
    config = configparser.ConfigParser()
    config.read(config_path)
    config.set('source0', 'uri', f'file://{os.path.abspath(video_path)}')
    
    # Create a temporary configuration file with updated paths
    temp_config_path = 'temp_config.txt'
    with open(temp_config_path, 'w') as configfile:
        config.write(configfile)
    
    # Create Pipeline element
    print("Creating Pipeline")
    pipeline = Gst.parse_launch(
        "uridecodebin name=source ! "
        "nvstreammux name=streammux ! "
        "nvinfer name=primary-gie ! "
        "nvtracker name=tracker ! "
        "nvinfer name=secondary-gie ! "
        "nvdsosd name=osd ! "
        "nvvideoconvert ! "
        "nvv4l2h264enc ! "
        "h264parse ! "
        "qtmux ! "
        "filesink name=sink"
    )
    
    PIPELINE = pipeline
    
    # Set properties from configuration file
    source = pipeline.get_by_name("source")
    source.set_property("uri", f"file://{os.path.abspath(video_path)}")
    
    streammux = pipeline.get_by_name("streammux")
    streammux.set_property("width", 1280)
    streammux.set_property("height", 720)
    streammux.set_property("batch-size", 1)
    streammux.set_property("batched-push-timeout", 40000)
    
    primary_gie = pipeline.get_by_name("primary-gie")
    primary_gie.set_property("config-file-path", "models/yoloe_config.txt")
    
    tracker = pipeline.get_by_name("tracker")
    tracker.set_property("ll-config-file", "models/tracker_config.yml")
    tracker.set_property("ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
    
    secondary_gie = pipeline.get_by_name("secondary-gie")
    secondary_gie.set_property("config-file-path", "models/rfdetr_config.txt")
    
    sink = pipeline.get_by_name("sink")
    sink.set_property("location", os.path.abspath(args.output))
    
    print(f"Playing file: {video_path}")
    print(f"Output will be saved to: {args.output}")
    
    # Create an event loop and feed GStreamer bus messages to it
    LOOP = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, LOOP)
    
    # Add probe to get metadata
    osdsinkpad = pipeline.get_by_name("osd").get_static_pad("sink")
    if not osdsinkpad:
        print("Unable to get sink pad of nvosd")
    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)
    
    # Start playing
    pipeline.set_state(Gst.State.PLAYING)
    try:
        LOOP.run()
    except:
        pass
    
    # Clean up
    pipeline.set_state(Gst.State.NULL)
    
    # Clean up temporary config file
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)
    
    print(f"Pipeline completed. Output saved to: {args.output}")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:])) 