# Raspberry Pi 5 OCR Bell Timer Setup Guide

This project uses OCR to recognize numbers on a digital clock and rings a buzzer/bell at a specified time.

## Hardware Requirements

- Raspberry Pi 5
- USB Camera or CSI Camera
- Buzzer/Speaker Module (5V)
- GPIO pins (default: GPIO 17)
- Power supply

## Installation

### 1. Install System Dependencies

```bash
sudo apt update
sudo apt install -y python3-pip
sudo apt install -y libatlas-base-dev libjasper-dev libtiff5 libjasper1 libharfbuzz0b libwebp6
sudo apt install -y tesseract-ocr
sudo apt install -y libopenjp2-7 libtiff6
```

### 2. Install Python Dependencies

```bash
pip3 install -r requirements.txt
```

### 3. Enable GPIO Access

```bash
sudo usermod -a -G gpio $USER
```

You may need to log out and back in for this to take effect.

## GPIO Wiring

Connect your buzzer/bell module to:
- **GPIO 17** (adjustable with --pin flag)
- **GND** (Ground)

If using a different GPIO pin, specify it when running:
```bash
python3 bell_timer.py --time 14:30 --pin 27
```

## Usage

### Using Live Camera Feed

```bash
python3 bell_timer.py --time 14:30
```

This will:
- Open the camera and display the live feed
- Use OCR to recognize the time from the digital clock display
- Ring the buzzer when the displayed time matches 14:30
- Press 'q' to quit

### Using Image File

```bash
python3 bell_timer.py --time 14:30 --image /path/to/clock_image.jpg
```

### Command Line Options

- `--time HH:MM` (required): Target time in 24-hour format
- `--pin PIN` (optional, default: 17): GPIO pin number for buzzer
- `--camera INDEX` (optional, default: 0): Camera device index
- `--image PATH` (optional): Path to image file instead of using live camera

## Troubleshooting

### Camera Issues
- Check if camera is enabled: `vcgencmd get_camera`
- List available cameras: `ls /dev/video*`
- Try different camera index: `--camera 0` or `--camera 1`

### OCR Not Working
- Ensure tesseract is installed: `tesseract --version`
- Check image quality - better lighting and focus improve accuracy
- Adjust thresholding in `extract_time_from_image()` if needed

### GPIO Permission Denied
- Run with sudo: `sudo python3 bell_timer.py --time 14:30`
- Or add user to gpio group: `sudo usermod -a -G gpio $USER`

### Buzzer Not Sounding
- Verify GPIO pin connection with: `python3 -c "from gpiozero import Buzzer; b = Buzzer(17); b.on(); input(); b.off()"`
- Check if positive wire is connected to GPIO pin and negative to GND

## Tips for Best OCR Accuracy

1. **Lighting**: Ensure bright, even lighting on the clock display
2. **Focus**: Position camera at a perpendicular angle to the screen
3. **Distance**: Clock should fill most of the frame
4. **Contrast**: Digital clock displays work best with high contrast
5. **Stability**: Mount camera on a tripod or fixed position

## Running at Boot

To run automatically on startup, add to crontab:

```bash
crontab -e
```

Add this line:
```
@reboot /usr/bin/python3 /path/to/bell_timer.py --time 14:30
```

Or create a systemd service file for better control.

## Notes

- The OCR check runs every 5 frames to conserve CPU resources
- Bell rings for 3 seconds by default (adjustable in code)
- Ensure the target time format is correct: HH:MM (24-hour format)
