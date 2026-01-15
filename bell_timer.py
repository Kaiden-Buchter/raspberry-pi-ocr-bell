#!/usr/bin/env python3
"""
Raspberry Pi 5 OCR Bell Timer
Recognizes numbers on a digital clock display using OCR and rings a bell at a specified time.
"""

import cv2
import pytesseract
import threading
import time
from datetime import datetime
from gpiozero import Buzzer
from PIL import Image
import re
import argparse

class OCRBellTimer:
    def __init__(self, target_time, gpio_pin=17, camera_index=0):
        """
        Initialize the OCR Bell Timer
        
        Args:
            target_time (str): Time to trigger bell in HH:MM format (e.g., "14:30")
            gpio_pin (int): GPIO pin number for the buzzer
            camera_index (int): Camera device index (0 for default)
        """
        self.target_time = target_time
        self.gpio_pin = gpio_pin
        self.camera_index = camera_index
        self.buzzer = Buzzer(gpio_pin)
        self.running = True
        self.bell_triggered = False
        
        try:
            pytesseract.pytesseract.pytesseract_cmd = r'/usr/bin/tesseract'
        except:
            pass
    
    def extract_time_from_image(self, image):
        """
        Extract time digits from camera image using OCR
        
        Args:
            image: OpenCV image frame
            
        Returns:
            str: Extracted time string in HH:MM format or None
        """
        try:
            # Preprocess image for better OCR
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            
            # Upscale for better OCR accuracy
            height, width = thresh.shape
            thresh = cv2.resize(thresh, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
            
            # Apply morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Perform OCR
            text = pytesseract.image_to_string(thresh, config='--psm 6 -c tessedit_char_whitelist=0123456789:')
            
            # Extract only the time portion (HH:MM)
            time_pattern = r'\d{1,2}:\d{2}'
            match = re.search(time_pattern, text)
            
            if match:
                return match.group(0)
            
            return None
        except Exception as e:
            print(f"Error extracting time from image: {e}")
            return None
    
    def ring_bell(self, duration=3):
        """
        Ring the buzzer/bell
        
        Args:
            duration (int): Duration in seconds
        """
        print(f"🔔 BELL RINGING! Time reached: {self.target_time}")
        self.buzzer.on()
        time.sleep(duration)
        self.buzzer.off()
        print("Bell stopped.")
    
    def process_camera_feed(self):
        """
        Continuously process camera feed and check for target time
        """
        cap = cv2.VideoCapture(self.camera_index)
        
        if not cap.isOpened():
            print(f"Error: Cannot open camera {self.camera_index}")
            return
        
        print(f"Camera opened successfully. Looking for time: {self.target_time}")
        print("Press 'q' to quit...")
        
        frame_count = 0
        check_frequency = 5  # Check OCR every N frames to save resources
        
        try:
            while self.running:
                ret, frame = cap.read()
                
                if not ret:
                    print("Error reading frame")
                    break
                
                # Display the frame
                cv2.imshow('OCR Bell Timer', frame)
                
                # Process every Nth frame for OCR
                frame_count += 1
                if frame_count % check_frequency == 0:
                    recognized_time = self.extract_time_from_image(frame)
                    
                    if recognized_time:
                        print(f"Recognized time: {recognized_time}", end='\r')
                        
                        # Check if time matches target
                        if recognized_time == self.target_time and not self.bell_triggered:
                            self.bell_triggered = True
                            # Ring bell in separate thread to not block camera processing
                            bell_thread = threading.Thread(target=self.ring_bell, args=(3,))
                            bell_thread.start()
                
                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
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
            
            if recognized_time == self.target_time:
                print("Target time matched!")
                self.ring_bell(3)
            else:
                print(f"Time {recognized_time} does not match target {self.target_time}")
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


def main():
    parser = argparse.ArgumentParser(description='Raspberry Pi OCR Bell Timer')
    parser.add_argument('--time', required=True, help='Target time in HH:MM format (e.g., 14:30)')
    parser.add_argument('--pin', type=int, default=17, help='GPIO pin for buzzer (default: 17)')
    parser.add_argument('--camera', type=int, default=0, help='Camera device index (default: 0)')
    parser.add_argument('--image', help='Image file path (if not using live camera)')
    
    args = parser.parse_args()
    
    # Validate time format
    try:
        datetime.strptime(args.time, '%H:%M')
    except ValueError:
        print("Error: Time must be in HH:MM format (e.g., 14:30)")
        return
    
    timer = OCRBellTimer(args.time, gpio_pin=args.pin, camera_index=args.camera)
    
    use_camera = args.image is None
    timer.start(use_camera=use_camera, image_path=args.image)


if __name__ == '__main__':
    main()
