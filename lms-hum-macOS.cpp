#include <iostream>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <unistd.h>
#include <termios.h>
#include <AudioToolbox/AudioToolbox.h>

// Interactive CoreAudio host for the LMS 60 Hz hum canceller (lms-hum.mlir).
//
// The kernel renders one second of  d[n] = sin(2*pi*440*t) + 0.7*sin(2*pi*60*t)
// with the 60 Hz mains hum adaptively subtracted by a 32-tap LMS filter whose
// weights start at zero. So each freshly rendered buffer starts buzzy and the
// hum fades out as the weights converge -- looping the buffer lets you hear that
// fade repeat. The arrow keys nudge the LMS step size @mu: larger mu converges
// faster (buzz clears sooner) but too large diverges.

const double SAMPLE_RATE = 44100.0;

// The kernel writes exactly one second: memref<44100xf64>. 440 Hz and 60 Hz
// both complete a whole number of cycles over this span (440 and 60), so the
// buffer loops with no discontinuity and needs no tail folding.
const size_t BATCH_SAMPLES = 44100;
const size_t LOOP_SAMPLES = 44100;

//===----------------------------------------------------------------------===//
// DSP-MLIR kernel boundary
//
// `run(out)` is lowered with `llvm.emit_c_interface`, so the backend emits
// `_mlir_ciface_run`, taking a pointer to a 1-D memref descriptor mirroring
// mlir::StridedMemRefType<double, 1>.
//===----------------------------------------------------------------------===//
struct MemRefDescriptor1D {
    double *basePtr;  // allocated pointer
    double *data;     // aligned pointer
    int64_t offset;
    int64_t size;     // sizes[0]
    int64_t stride;   // strides[0]
};

extern "C" void _mlir_ciface_run(MemRefDescriptor1D *out);
extern "C" double mu;  // the DSL's @mu global (LMS step size, fixed here)
extern "C" double wet; // the DSL's @wet global: how much of the hum estimate to
                       // subtract (0 = full hum, 1 = hum removed) -- the knob

// Render one batch by invoking the kernel, which reads the current `mu` global.
static void fillBatch(double *buf, size_t n) {
    MemRefDescriptor1D desc{buf, buf, 0, static_cast<int64_t>(n), 1};
    _mlir_ciface_run(&desc);
}

// Double-buffered batch: the audio thread streams from gActive while the render
// thread rebuilds the next batch into the back buffer, then swaps.
static double gBufferA[BATCH_SAMPLES];
static double gBufferB[BATCH_SAMPLES];
static std::atomic<double *> gActive{gBufferA};
static double *gBackBuffer = gBufferB;
static size_t gReadPos = 0; // only touched by the audio thread
static std::atomic<bool> gRunning{true};

// How often the render thread rebuilds the batch. This bounds how quickly a mu
// change becomes audible.
constexpr auto REGEN_PERIOD = std::chrono::milliseconds(100);

// Rebuild the back buffer from the kernel (which reads the current `mu` global)
// and publish it. Only the render thread calls this once audio is live, so it
// is the sole owner of gBackBuffer.
static void regenerate() {
    double *back = gBackBuffer;
    fillBatch(back, BATCH_SAMPLES);
    gBackBuffer = gActive.exchange(back, std::memory_order_acq_rel);
}

// Background render thread: periodically regenerate so mu changes written from
// the keyboard thread are picked up asynchronously.
static void renderLoop() {
    while (gRunning.load(std::memory_order_relaxed)) {
        regenerate();
        std::this_thread::sleep_for(REGEN_PERIOD);
    }
}

// Raw terminal input.
struct termios orig_termios;

void disableRawMode() {
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
}

void enableRawMode() {
    tcgetattr(STDIN_FILENO, &orig_termios);
    atexit(disableRawMode);
    struct termios raw = orig_termios;
    raw.c_lflag &= ~(ECHO | ICANON);
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
}

// CoreAudio render callback.
OSStatus AudioCallback(void *inRefCon,
                       AudioUnitRenderActionFlags *ioActionFlags,
                       const AudioTimeStamp *inTimeStamp,
                       UInt32 inBusNumber,
                       UInt32 inNumberFrames,
                       AudioBufferList *ioData)
{
    float *leftChannel = static_cast<float*>(ioData->mBuffers[0].mData);
    float *rightChannel = static_cast<float*>(ioData->mBuffers[1].mData);

    const double *batch = gActive.load(std::memory_order_acquire);
    size_t pos = gReadPos;

    for (UInt32 i = 0; i < inNumberFrames; ++i) {
        float s = static_cast<float>(batch[pos]); // f64 -> f32 on copy
        leftChannel[i] = s;
        rightChannel[i] = s;
        if (++pos >= LOOP_SAMPLES)
            pos = 0;
    }

    gReadPos = pos;
    return noErr;
}

int main() {
    std::cout << "Initializing macOS Core Audio Engine..." << std::endl;

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

    AudioUnit outputUnit;
    if (AudioComponentInstanceNew(comp, &outputUnit) != noErr) {
        std::cerr << "Failed to open Audio Unit instance." << std::endl;
        return 1;
    }

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
                             0,
                             &streamFormat,
                             sizeof(streamFormat)) != noErr) {
        std::cerr << "Failed to apply basic audio stream format settings." << std::endl;
        return 1;
    }

    AURenderCallbackStruct callbackStruct;
    callbackStruct.inputProc = AudioCallback;
    callbackStruct.inputProcRefCon = nullptr;

    if (AudioUnitSetProperty(outputUnit,
                             kAudioUnitProperty_SetRenderCallback,
                             kAudioUnitScope_Input,
                             0,
                             &callbackStruct,
                             sizeof(callbackStruct)) != noErr) {
        std::cerr << "Failed to register audio render callback structure loop." << std::endl;
        return 1;
    }

    if (AudioUnitInitialize(outputUnit) != noErr) {
        std::cerr << "Could not initialize Core Audio hardware context buffers." << std::endl;
        return 1;
    }

    // Fix mu small so the persistent weights converge over a few looped buffers
    // and stay converged (keeps the 440 Hz tone undistorted). The interactive
    // control is `wet`, not mu. Start fully wet (hum removed).
    mu = 0.001;
    wet = 1.0;
    regenerate();

    if (AudioOutputUnitStart(outputUnit) != noErr) {
        std::cerr << "Could not start audio stream output pipeline." << std::endl;
        return 1;
    }

    std::thread renderThread(renderLoop);

    std::cout << "\n==============================================" << std::endl;
    std::cout << "   CoreAudio LMS Hum Canceller Running!        " << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << " -> 440 Hz tone with 60 Hz mains hum; the LMS filter estimates" << std::endl;
    std::cout << "    the hum and the knob below sweeps how much of it to subtract." << std::endl;
    std::cout << " -> Press [UP Arrow]   to add cancellation (sweep toward hum-free)" << std::endl;
    std::cout << " -> Press [DOWN Arrow] to back it off (sweep the 60 Hz hum back in)" << std::endl;
    std::cout << " -> Press [Q] or Ctrl+C to stop the program safely" << std::endl;
    std::cout << "==============================================\n" << std::endl;

    enableRawMode();

    char c;
    while (read(STDIN_FILENO, &c, 1) == 1 && c != 'q' && c != 'Q') {
        // Arrow keys send: '\x1b', '[', then 'A' (Up) / 'B' (Down).
        if (c == '\x1b') {
            char seq[2];
            if (read(STDIN_FILENO, &seq[0], 1) == 1 && read(STDIN_FILENO, &seq[1], 1) == 1) {
                if (seq[0] == '[') {
                    // Arrow keys only nudge the `mu` symbol; the render thread
                    // picks up the new value on its next pass. Clamp to a stable
                    // range: too large a step size makes the 32-tap LMS diverge.
                    if (seq[1] == 'A') { // Up Arrow
                        wet = std::min(1.0, wet + 0.05);
                        printf("\rHum cancellation: %3.0f%%   ", wet * 100.0);
                        fflush(stdout);
                    } else if (seq[1] == 'B') { // Down Arrow
                        wet = std::max(0.0, wet - 0.05);
                        printf("\rHum cancellation: %3.0f%%   ", wet * 100.0);
                        fflush(stdout);
                    }
                }
            }
        }
    }

    std::cout << "\n\nStopping audio engine and cleaning up channels..." << std::endl;
    gRunning.store(false, std::memory_order_relaxed);
    renderThread.join();
    AudioOutputUnitStop(outputUnit);
    AudioComponentInstanceDispose(outputUnit);
    return 0;
}
