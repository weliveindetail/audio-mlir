#include <iostream>
#include <cmath>
#include <vector>
#include <unistd.h>
#include <termios.h>
#include <AudioToolbox/AudioToolbox.h>

// Global DSP State Constants
const double SAMPLE_RATE = 44100.0;
const double FREQUENCY = 440.0; // Base oscillator frequency (A4)

// Atomic or simple globals since they are accessed on the real-time audio thread
struct DSPState {
    double globalTime = 0.0;
    double lastFilteredValue = 0.0;
    double cutoffFreq = 1000.0; // Initial cut-off
} gState;

// Raw Terminal Input Configurations
struct termios orig_termios;

void disableRawMode() {
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
}

void enableRawMode() {
    tcgetattr(STDIN_FILENO, &orig_termios);
    atexit(disableRawMode);
    
    struct termios raw = orig_termios;
    raw.c_lflag &= ~(ECHO | ICANON); // Turn off automatic echo and canonical line buffering
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
}

// macOS Core Audio Render Callback Function
OSStatus AudioCallback(void *inRefCon,
                       AudioUnitRenderActionFlags *ioActionFlags,
                       const AudioTimeStamp *inTimeStamp,
                       UInt32 inBusNumber,
                       UInt32 inNumberFrames,
                       AudioBufferList *ioData) 
{
    DSPState* state = static_cast<DSPState*>(inRefCon);
    
    // Core Audio delivers Float32 linear PCM data buffers
    float *leftChannel = static_cast<float*>(ioData->mBuffers[0].mData);
    float *rightChannel = static_cast<float*>(ioData->mBuffers[1].mData);
    
    // Dynamic Filter Coefficient Calculations (First-order Low-pass IIR)
    double RC = 1.0 / (2.0 * M_PI * state->cutoffFreq);
    double dt = 1.0 / SAMPLE_RATE;
    double alpha = dt / (RC + dt);
    double currentVal = state->lastFilteredValue;
    
    for (UInt32 i = 0; i < inNumberFrames; ++i) {
        double t = state->globalTime + (static_cast<double>(i) / SAMPLE_RATE);
        
        // 1. Generate Mathematical Sawtooth Wave
        double phase = t * FREQUENCY;
        double signal = 2.0 * (phase - std::floor(phase + 0.5));
        
        // 2. Apply Low-Pass Filter
        currentVal = currentVal + alpha * (signal - currentVal);
        
        // 3. Output safely to Stereo channels (-1.0 to 1.0 Float32 range)
        leftChannel[i] = static_cast<float>(currentVal);
        rightChannel[i] = static_cast<float>(currentVal);
    }
    
    // Maintain state counters across discrete audio processing blocks
    state->globalTime += (static_cast<double>(inNumberFrames) / SAMPLE_RATE);
    state->lastFilteredValue = currentVal;
    
    return noErr;
}

int main() {
    std::cout << "Initializing macOS Core Audio Engine..." << std::endl;
    
    // 1. Describe the Audio Component (Default Output Hardware Device)
    AudioComponentDescription desc;
    desc.componentType = kAudioUnitType_Output;
    desc.componentSubType = kAudioUnitSubType_DefaultOutput;
    desc.componentManufacturer = kAudioUnitManufacturer_Apple;
    desc.componentFlags = 0;
    desc.componentFlagsMask = 0;
    
    AudioComponent comp = AudioComponentFindNext(NULL, &desc);
    if (!comp) {
        std::cerr << "Failed to locate native Mac default audio output hardware device." << std::endl;
        return 1;
    }
    
    // 2. Create the Audio Unit Instance
    AudioUnit outputUnit;
    if (AudioComponentInstanceNew(comp, &outputUnit) != noErr) {
        std::cerr << "Failed to open Audio Unit instance." << std::endl;
        return 1;
    }
    
    // 3. Define the Stream Format (Stereo, Non-interleaved, 32-bit Float PCM)
    AudioStreamBasicDescription streamFormat;
    streamFormat.mSampleRate = SAMPLE_RATE;
    streamFormat.mFormatID = kAudioFormatLinearPCM;
    streamFormat.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsNonInterleaved;
    streamFormat.mBytesPerPacket = 4;
    streamFormat.mFramesPerPacket = 1;
    streamFormat.mBytesPerFrame = 4;
    streamFormat.mChannelsPerFrame = 2; // Stereo
    streamFormat.mBitsPerChannel = 32;  // Float32
    streamFormat.mReserved = 0;
    
    if (AudioUnitSetProperty(outputUnit,
                             kAudioUnitProperty_StreamFormat,
                             kAudioUnitScope_Input,
                             0, // Output element 0 input scope
                             &streamFormat,
                             sizeof(streamFormat)) != noErr) {
        std::cerr << "Failed to apply basic audio stream format settings." << std::endl;
        return 1;
    }
    
    // 4. Register the Real-time Callback Render Node
    AURenderCallbackStruct callbackStruct;
    callbackStruct.inputProc = AudioCallback;
    callbackStruct.inputProcRefCon = &gState;
    
    if (AudioUnitSetProperty(outputUnit,
                             kAudioUnitProperty_SetRenderCallback,
                             kAudioUnitScope_Input,
                             0,
                             &callbackStruct,
                             sizeof(callbackStruct)) != noErr) {
        std::cerr << "Failed to register audio render callback structure loop." << std::endl;
        return 1;
    }
    
    // 5. Initialize and Boot up the Core Audio Stream
    if (AudioUnitInitialize(outputUnit) != noErr) {
        std::cerr << "Could not initialize Core Audio hardware context buffers." << std::endl;
        return 1;
    }
    
    if (AudioOutputUnitStart(outputUnit) != noErr) {
        std::cerr << "Could not start audio stream output pipeline." << std::endl;
        return 1;
    }
    
    std::cout << "\n==============================================" << std::endl;
    std::cout << "   CoreAudio Low-Pass Filter Synth Running!   " << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << " -> Press [UP Arrow]   to raise the cut-off frequency" << std::endl;
    std::cout << " -> Press [DOWN Arrow] to lower the cut-off frequency" << std::endl;
    std::cout << " -> Press [Q] or Ctrl+C to stop the program safely" << std::endl;
    std::cout << "==============================================\n" << std::endl;
    
    // Enable unbuffered raw terminal input processing
    enableRawMode();
    
    // 6. Interactive Command-Line Run Loop
    char c;
    while (read(STDIN_FILENO, &c, 1) == 1 && c != 'q' && c != 'Q') {
        // macOS terminal arrow keys send a multi-byte escape sequence: '\x1b', '[', followed by 'A' (Up) or 'B' (Down)
        if (c == '\x1b') {
            char seq[2];
            if (read(STDIN_FILENO, &seq[0], 1) == 1 && read(STDIN_FILENO, &seq[1], 1) == 1) {
                if (seq[0] == '[') {
                    if (seq[1] == 'A') { // Up Arrow
                        gState.cutoffFreq = std::min(15000.0, gState.cutoffFreq + 100.0);
                        printf("\rCut-off Frequency: %.1f Hz   ", gState.cutoffFreq);
                        fflush(stdout);
                    } else if (seq[1] == 'B') { // Down Arrow
                        gState.cutoffFreq = std::max(40.0, gState.cutoffFreq - 100.0);
                        printf("\rCut-off Frequency: %.1f Hz   ", gState.cutoffFreq);
                        fflush(stdout);
                    }
                }
            }
        }
    }
    
    // 7. Cleanup Resources Safely on Exit
    std::cout << "\n\nStopping audio engine and cleaning up channels..." << std::endl;
    AudioOutputUnitStop(outputUnit);
    AudioComponentInstanceDispose(outputUnit);
    return 0;
}
