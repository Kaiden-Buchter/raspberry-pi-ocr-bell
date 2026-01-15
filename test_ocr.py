#!/usr/bin/env python3
"""
Test script for OCR functionality without requiring GPIO
Useful for testing OCR accuracy on image files
"""

import cv2
import pytesseract
import argparse
from PIL import Image
import re

def test_ocr_on_image(image_path):
    """
    Test OCR on a single image file
    
    Args:
        image_path (str): Path to image file
    """
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Cannot read image from {image_path}")
        return
    
    print(f"Image dimensions: {image.shape}")
    
    # Preprocess image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    height, width = thresh.shape
    thresh = cv2.resize(thresh, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Save preprocessed image for inspection
    cv2.imwrite('/tmp/preprocessed.jpg', thresh)
    print("Saved preprocessed image to /tmp/preprocessed.jpg")
    
    # Run OCR
    print("\nRunning OCR...")
    text = pytesseract.image_to_string(thresh, config='--psm 6 -c tessedit_char_whitelist=0123456789:')
    print(f"Raw OCR output: '{text}'")
    
    # Extract time
    time_pattern = r'\d{1,2}:\d{2}'
    match = re.search(time_pattern, text)
    
    if match:
        recognized_time = match.group(0)
        print(f"Recognized time: {recognized_time}")
    else:
        print("Could not extract time from OCR output")

def main():
    parser = argparse.ArgumentParser(description='Test OCR on image files')
    parser.add_argument('image', help='Path to image file')
    
    args = parser.parse_args()
    test_ocr_on_image(args.image)

if __name__ == '__main__':
    main()
