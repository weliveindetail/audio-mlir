#!/usr/bin/env python

import math
import numpy as np
import pygame

# Initialize Pygame and Mixer
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=1024)

# Detect the precise hardware channel architecture initialized by your OS
actual_freq, actual_format, num_channels = pygame.mixer.get_init()

# Create a small window to capture up/down key presses
screen = pygame.display.set_mode((400, 200))
pygame.display.set_caption("DSP Filter | Use Up/Down Arrow Keys")

# DSP Configuration
SAMPLE_RATE = actual_freq
FREQUENCY = 440.0  # Base oscillator frequency (A4)
cutoff_freq = 1000.0  # Starting low-pass cut-off frequency

# Setup continuous streaming buffer
BUFFER_SAMPLES = 2048  # Small buffer size for low latency response
stream_array = np.zeros((BUFFER_SAMPLES, num_channels) if num_channels > 1 else BUFFER_SAMPLES, dtype=np.int16)
sound = pygame.sndarray.make_sound(stream_array)

# Continuous tracking variables across blocks
global_phase = 0.0
last_filtered_value = 0.0

def update_audio_buffer():
    """Generates the next DSP chunk and injects it straight into the active buffer."""
    global global_phase, last_filtered_value, cutoff_freq
    
    # Generate timeframe for this chunk
    t = (np.arange(BUFFER_SAMPLES) + global_phase) / SAMPLE_RATE
    global_phase += BUFFER_SAMPLES  # Advance phase to avoid audio clicks
    
    # 1. Generate Raw Sawtooth Wave
    phase = t * FREQUENCY
    signal = 2.0 * (phase - np.floor(phase + 0.5))
    
    # 2. Calculate dynamic IIR Low-Pass filter coefficients
    RC = 1.0 / (2.0 * math.pi * cutoff_freq)
    dt = 1.0 / SAMPLE_RATE
    alpha = dt / (RC + dt)
    
    # 3. Apply the filter sequentially to preserve state history
    filtered = np.zeros_like(signal)
    current_val = last_filtered_value
    for i in range(BUFFER_SAMPLES):
        current_val = current_val + alpha * (signal[i] - current_val)
        filtered[i] = current_val
    last_filtered_value = current_val  # Store state for the next frame block
    
    # 4. Normalize audio and scale to 16-bit range
    audio_data = np.clip(filtered, -1.0, 1.0) * 32767
    audio_16bit = audio_data.astype(np.int16)
    
    # 5. Format to correct channel shape and write into sound memory
    if num_channels > 1:
        stereo_data = np.repeat(audio_16bit.reshape(-1, 1), num_channels, axis=1)
        pygame.sndarray.samples(sound)[:] = stereo_data
    else:
        pygame.sndarray.samples(sound)[:] = audio_16bit

# Start looping the buffer sound continuously
sound.play(-1)

# Main Application Run Loop
running = True
clock = pygame.time.Clock()

print(f"System Audio Mode Detected: {num_channels} Channel(s) @ {SAMPLE_RATE}Hz")
print(f"Initial Cut-off: {cutoff_freq} Hz")

try:
    while running:
        # Update audio data continuously at the speed of the application execution
        update_audio_buffer()
        
        # Process keyboard input events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    cutoff_freq = min(10000.0, cutoff_freq + 100)  # Raise cut-off
                    print(f"Cut-off Frequency: {cutoff_freq:.1f} Hz")
                elif event.key == pygame.K_DOWN:
                    cutoff_freq = max(60.0, cutoff_freq - 100)   # Lower cut-off
                    print(f"Cut-off Frequency: {cutoff_freq:.1f} Hz")

        # Limit loop frequency to avoid excessive CPU strain
        clock.tick(60)

except KeyboardInterrupt:
    # Exit with Ctrl+C
    print("\nExiting...")

finally:
    pygame.quit()
