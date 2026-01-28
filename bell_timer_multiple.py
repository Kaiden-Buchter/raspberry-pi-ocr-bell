#!/usr/bin/env python3
"""
Raspberry Pi 5 OCR Bell Timer - Multiple Times Version
Recognizes numbers on a digital clock and rings a bell at multiple specified times.
"""

import cv2
import pytesseract
import threading
import time
import json
from datetime import datetime
from PIL import Image
import re
import argparse
import os
from play_bell_sound import play_bell_sound

class OCRBellTimerMultiple:
    def __init__(self, target_times, gpio_pin=17, camera_index=0):
        """
        Initialize the OCR Bell Timer for multiple times
        
        Args:
            target_times (list): List of times in HH:MM format
            camera_index (int): Camera device index (0 for default)
        """
        self.target_times = target_times
        self.camera_index = camera_index
        self.running = True
        self.triggered_times = set()  # Track which times have already triggered
        
        try:
            pytesseract.pytesseract.pytesseract_cmd = r'/usr/bin/tesseract'
        except:
            pass
        
        print(f"Target times for bell: {', '.join(self.target_times)}")
    
    def extract_time_from_image(self, image):
        """
        Extract time digits from camera image using OCR, focusing on red digits and cropping to clock face.
        Tries multiple Tesseract PSM modes and allows for tighter cropping.
        Args:
            image: OpenCV image frame
        Returns:
            str: Extracted time string in HH:MM format or None
        """
        try:
            # --- Step 1: Crop to clock face area (tighter crop) ---
            h, w, _ = image.shape
            # Try a tighter crop (adjust these as needed)
            crop_x1 = int(w * 0.28)
            crop_x2 = int(w * 0.72)
            crop_y1 = int(h * 0.36)
            crop_y2 = int(h * 0.64)
            cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]

            # --- Step 2: Isolate red digits ---
            hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
            lower_red1 = (0, 70, 50)
            upper_red1 = (10, 255, 255)
            lower_red2 = (170, 70, 50)
            upper_red2 = (180, 255, 255)
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
            red_digits = cv2.bitwise_and(cropped, cropped, mask=mask)

            # --- Step 3: Convert to grayscale and enhance ---
            gray = cv2.cvtColor(red_digits, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            gray = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
            gray = cv2.bitwise_not(gray)
            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

            # --- Step 4: Upscale for better OCR accuracy ---
            height, width = thresh.shape
            thresh = cv2.resize(thresh, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            # --- Step 5: Try multiple Tesseract PSM modes ---
            configs = [
                '--psm 13 -c tessedit_char_whitelist=0123456789:',
                '--psm 6 -c tessedit_char_whitelist=0123456789:',
                '--psm 7 -c tessedit_char_whitelist=0123456789:'
            ]
            time_pattern = r'\d{1,2}:\d{2}'
            for config in configs:
                text = pytesseract.image_to_string(thresh, config=config)
                match = re.search(time_pattern, text)
                if match:
                    return match.group(0)
            return None
        except Exception as e:
            print(f"Error extracting time from image: {e}")
            return None
    
    def ring_bell(self, trigger_time, duration=3, sound_file="bell.wav"):
        """
        Ring the buzzer/bell and play a sound file
        
        Args:
            trigger_time (str): The time that triggered the bell
            duration (int): Duration in seconds
            sound_file (str): Path to bell sound file (wav or mp3)
        """
        print(f"\n🔔 BELL RINGING! Time reached: {trigger_time}")
        # Play sound in a separate thread so it doesn't block
        threading.Thread(target=play_bell_sound, args=(sound_file,), daemon=True).start()
        time.sleep(duration)
        print("Bell stopped.\n")
    
    def process_camera_feed(self):
        """
        Continuously process camera feed and check for all target times, without opening a camera preview window.
        Only prints when an image is read and what text was found.
        """
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"Error: Cannot open camera {self.camera_index}")
            return
        print(f"Camera opened successfully. Watching for times: {', '.join(self.target_times)}")
        print("Press Ctrl+C to quit...\n")

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("Error reading frame")
                    break

                # Save the frame as an image file
                image_path = "temp_capture.jpg"
                cv2.imwrite(image_path, frame)

                # Read the image back and process OCR
                image = cv2.imread(image_path)
                recognized_time = self.extract_time_from_image(image)
                print(f"[Image read] OCR detected: {recognized_time}")
                if recognized_time:
                    if recognized_time in self.target_times and recognized_time not in self.triggered_times:
                        print(f"Target time matched: {recognized_time}")
                        self.triggered_times.add(recognized_time)
                        bell_thread = threading.Thread(target=self.ring_bell, args=(recognized_time, 3))
                        bell_thread.start()

                # Delete the image file
                try:
                    os.remove(image_path)
                except Exception as e:
                    print(f"Warning: Could not delete temp image: {e}")

                # Wait 1 second before next capture
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nExiting on user request.")
        finally:
            cap.release()
            self.running = False
    
    def process_image_file(self, image_path):
        """
        Process a single image file instead of live camera feed
        
        Args:
            image_path (str): Path to image file
        """
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error: Cannot read image from {image_path}")
            return
        
        recognized_time = self.extract_time_from_image(image)
        
        if recognized_time:
            print(f"Recognized time: {recognized_time}")
            
            if recognized_time in self.target_times:
                print(f"✓ Time {recognized_time} matches a target time!")
                self.ring_bell(recognized_time, 3)
            else:
                print(f"✗ Time {recognized_time} does not match any target times")
                print(f"   Target times: {', '.join(self.target_times)}")
        else:
            print("Could not recognize time from image")
    
    def start(self, use_camera=True, image_path=None):
        """
        Start the timer
        
        Args:
            use_camera (bool): If True, use live camera feed; if False, use image file
            image_path (str): Path to image file (required if use_camera is False)
        """
        if use_camera:
            self.process_camera_feed()
        else:
            if not image_path:
                print("Error: image_path required when use_camera=False")
                return
            self.process_image_file(image_path)


def load_config_file(config_path):
    """
    Load times from a JSON configuration file
    
    Args:
        config_path (str): Path to JSON config file
        
    Returns:
        list: List of times in HH:MM format
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        if 'times' not in config:
            print("Error: 'times' key not found in config file")
            return None
        
        times = config['times']
        if not isinstance(times, list):
            print("Error: 'times' must be a list")
            return None
        
        # Validate time format
        for t in times:
            try:
                datetime.strptime(t, '%H:%M')
            except ValueError:
                print(f"Error: Invalid time format '{t}'. Use HH:MM format.")
                return None
        
        return times
    
    except FileNotFoundError:
        print(f"Error: Config file not found: {config_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in config file: {config_path}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Raspberry Pi OCR Bell Timer - Multiple Times')
    parser.add_argument('--times', nargs='+', help='Target times in HH:MM format (e.g., 14:30 15:45)')
    parser.add_argument('--config', help='Path to JSON config file with times')
    parser.add_argument('--pin', type=int, default=17, help='GPIO pin for buzzer (default: 17)')
    parser.add_argument('--camera', type=int, default=0, help='Camera device index (default: 0)')
    parser.add_argument('--image', help='Image file path (if not using live camera)')
    
    args = parser.parse_args()
    
    # Get times from either --times or --config
    if args.config:
        target_times = load_config_file(args.config)
        if not target_times:
            return
    elif args.times:
        target_times = args.times
        # Validate time format
        for t in target_times:
            try:
                datetime.strptime(t, '%H:%M')
            except ValueError:
                print(f"Error: Time '{t}' must be in HH:MM format (e.g., 14:30)")
                return
    else:
        print("Error: Must provide either --times or --config")
        parser.print_help()
        return
    
    if not target_times:
        print("Error: No valid times specified")
        return
    
    timer = OCRBellTimerMultiple(target_times, gpio_pin=args.pin, camera_index=args.camera)
    
    use_camera = args.image is None
    timer.start(use_camera=use_camera, image_path=args.image)


if __name__ == '__main__':
    main()
