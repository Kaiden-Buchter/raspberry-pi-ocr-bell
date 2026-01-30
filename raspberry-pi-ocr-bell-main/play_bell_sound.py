# Download or provide a bell sound file (bell.wav or bell.mp3) in the project directory.
# This script will play the sound using aplay (for .wav) or mpg123 (for .mp3) when the bell rings.
# Make sure to install mpg123 if using mp3: sudo apt-get install mpg123

import os
import subprocess

def play_bell_sound(sound_file):
    if not os.path.exists(sound_file):
        print(f"Sound file not found: {sound_file}")
        return
    if sound_file.endswith('.wav'):
        subprocess.run(['aplay', sound_file])
    elif sound_file.endswith('.mp3'):
        subprocess.run(['mpg123', sound_file])
    else:
        print("Unsupported sound file format.")
