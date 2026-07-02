#!/usr/bin/env python

import math
import numpy as np
import pygame
import sounddevice as sd

# Initialize Pygame strictly for the GUI and Window Input
pygame.init()
screen = pygame.display.set_mode((450, 200))
pygame.display.set_caption("Clean DSP Filter | Use Up/Down Arrow Keys")
pygame.font.init()
font = pygame.font.SysFont("Arial", 24)

# Global DSP Configuration
SAMPLE_RATE = 44100
FREQUENCY = 440.0  # Base oscillator frequency (A4)
cutoff_freq = 1000.0  # Starting low-pass cut-off frequency

# Continuous phase trackers for the callback thread
global_time = 0.0
last_filtered_value = 0.0

def audio_callback(outdata, frames, time_info, status):
    """Modern background thread callback for flawless, click-free audio streaming."""
    global global_time, last_filtered_value, cutoff_freq
    
    if status:
        print(status)
        
    # 1. Vectorized time array generation
    t = global_time + np.arange(frames) / SAMPLE_RATE
    global_time += frames / SAMPLE_RATE  # Keep phase aligned perfectly
    
    # 2. Mathematical Sawtooth Generation
    phase = t * FREQUENCY
    signal = 2.0 * (phase - np.floor(phase + 0.5))
    
    # 3. Dynamic IIR Low-Pass filter coefficient calculation
    RC = 1.0 / (2.0 * math.pi * cutoff_freq)
    dt = 1.0 / SAMPLE_RATE
    alpha = dt / (RC + dt)
    
    # 4. Filter processing 
    filtered = np.zeros(frames, dtype=np.float32)
    current_val = last_filtered_value
    for i in range(frames):
        current_val = current_val + alpha * (signal[i] - current_val)
        filtered[i] = current_val
    last_filtered_value = current_val  
    
    # 5. Output Float32 values safely between -1.0 and 1.0 (eliminates clipping/truncation noise)
    # sounddevice splits this to Stereo automatically if outdata is 2D
    if outdata.shape[1] == 2:
        outdata[:] = np.repeat(filtered.reshape(-1, 1), 2, axis=1)
    else:
        outdata[:] = filtered.reshape(-1, 1)

# Start the dedicated low-latency audio stream using Float32
stream = sd.OutputStream(channels=2, callback=audio_callback, samplerate=SAMPLE_RATE, dtype='float32')
stream.start()

# Main UI/Keyboard Loop
running = True
clock = pygame.time.Clock()

try:
    while running:
        # Process keyboard input events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    cutoff_freq = min(15000.0, cutoff_freq + 100)  # Raise cut-off
                elif event.key == pygame.K_DOWN:
                    cutoff_freq = max(40.0, cutoff_freq - 100)   # Lower cut-off

        # GUI Visuals
        screen.fill((20, 24, 30))  # Sleek dark blue/gray
        title_surface = font.render("Sawtooth with Low-Pass Filter", True, (230, 230, 230))
        freq_surface = font.render(f"Cut-off Frequency: {cutoff_freq:.1f} Hz", True, (0, 220, 255))
        hint_surface = font.render("Press UP/DOWN arrows to sweep filter", True, (140, 145, 150))
        
        screen.blit(title_surface, (20, 30))
        screen.blit(freq_surface, (20, 80))
        screen.blit(hint_surface, (20, 140))
        
        pygame.display.flip()
        clock.tick(60)

except KeyboardInterrupt:
    print("\nExiting...")
finally:
    stream.stop()
    stream.close()
    pygame.quit()