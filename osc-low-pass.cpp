#include <iostream>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <unistd.h>
#include <termios.h>
#include <AudioToolbox/AudioToolbox.h>

// Global DSP State Constants
const double SAMPLE_RATE = 44100.0;

// The kernel convolves 44100 signal samples with a 101-tap FIR, so linear
// convolution yields 44100 + 101 - 1 = 44200 output samples. The out-buffer
// must match that length exactly (memref<44200xf64> on the MLIR side). Note the
// extra 100 tail samples mean the loop point is not a perfect 1 s boundary, so
// there is a faint seam when the batch wraps; acceptable for now.
const size_t BATCH_SAMPLES = 44200;

//===----------------------------------------------------------------------===//
// DSP-MLIR kernel boundary
//
// The DSL `run(out, fc)` is lowered with `llvm.emit_c_interface`, so the
// backend emits `_mlir_ciface_run`, which takes a pointer to a 1-D memref
// descriptor (destination-passing: the kernel writes into the caller's buffer)
// and the cut-off frequency. This mirrors mlir::StridedMemRefType<double, 1>.
//===----------------------------------------------------------------------===//
struct MemRefDescriptor1D {
    double *basePtr;  // allocated pointer
    double *data;     // aligned pointer
    int64_t offset;
    int64_t size;     // sizes[0]
    int64_t stride;   // strides[0]
};

extern "C" void _mlir_ciface_run(MemRefDescriptor1D *out, double cutoff);

// Render one batch into `buf` by invoking the DSP-MLIR kernel.
static void fillBatch(double *buf, size_t n, double cutoff) {
    MemRefDescriptor1D desc{buf, buf, 0, static_cast<int64_t>(n), 1};
    _mlir_ciface_run(&desc, cutoff);
}

// Double-buffered batch: the audio thread streams from gActive while the
// keyboard thread renders the next batch into the back buffer, then swaps.
static double gBufferA[BATCH_SAMPLES];
static double gBufferB[BATCH_SAMPLES];
static std::atomic<double *> gActive{gBufferA};
static double *gBackBuffer = gBufferB;
static size_t gReadPos = 0; // only touched by the audio thread
static double gCutoff = 1000.0; // current cut-off (keyboard thread + display)

// Render with the given cut-off into the back buffer and publish it.
static void regenerate(double cutoff) {
    double *back = gBackBuffer;
    fillBatch(back, BATCH_SAMPLES, cutoff);
    gBackBuffer = gActive.exchange(back, std::memory_order_acq_rel);
}

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
    // Core Audio delivers Float32 linear PCM data buffers
    float *leftChannel = static_cast<float*>(ioData->mBuffers[0].mData);
    float *rightChannel = static_cast<float*>(ioData->mBuffers[1].mData);

    // Stream from the currently published batch, converting f64 -> f32 and
    // looping at the end of the one-second buffer.
    const double *batch = gActive.load(std::memory_order_acquire);
    size_t pos = gReadPos;

    for (UInt32 i = 0; i < inNumberFrames; ++i) {
        float s = static_cast<float>(batch[pos]); // f64 -> f32 on copy
        leftChannel[i] = s;
        rightChannel[i] = s;
        if (++pos >= BATCH_SAMPLES)
            pos = 0;
    }

    gReadPos = pos;
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
    callbackStruct.inputProcRefCon = nullptr; // callback streams from global batch
    
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

    // Render the first batch with the initial cut-off before audio starts.
    regenerate(gCutoff);

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
                        gCutoff = std::min(15000.0, gCutoff + 100.0);
                        regenerate(gCutoff);
                        printf("\rCut-off Frequency: %.1f Hz   ", gCutoff);
                        fflush(stdout);
                    } else if (seq[1] == 'B') { // Down Arrow
                        gCutoff = std::max(40.0, gCutoff - 100.0);
                        regenerate(gCutoff);
                        printf("\rCut-off Frequency: %.1f Hz   ", gCutoff);
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
