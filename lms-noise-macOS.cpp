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

// Interactive CoreAudio host for the LMS adaptive noise canceller
// (lms-noise.mlir): a 440 Hz tone buried in broadband noise, with the noise
// adaptively subtracted by a 32-tap LMS filter.
//
//   Up / Down arrows  -> @wet:        how much of the estimated noise to remove
//                                     (0 = noise back in, 1 = clean tone).
//   Left / Right arrows -> @noise_kind: cycle the noise *color* the kernel
//                                     generates -- white / pink / brown / OU.
//
// The noise reference is generated in-kernel from an LCG stream, optionally
// colored by the pink/brown/OU recurrences (the same math the dsp.noise_*
// dialect ops encode); no RNG primitive or host-fed data is needed.

const double SAMPLE_RATE = 44100.0;

// The kernel writes exactly one second: memref<44100xf64>.
const size_t BATCH_SAMPLES = 44100;
const size_t LOOP_SAMPLES = 44100;

//===----------------------------------------------------------------------===//
// DSP-MLIR kernel boundary
//===----------------------------------------------------------------------===//
struct MemRefDescriptor1D {
    double *basePtr;  // allocated pointer
    double *data;     // aligned pointer
    int64_t offset;
    int64_t size;     // sizes[0]
    int64_t stride;   // strides[0]
};

extern "C" void _mlir_ciface_run(MemRefDescriptor1D *out);
extern "C" double mu;         // the DSL's @mu global (LMS step size, fixed here)
extern "C" double wet;        // @wet: how much of the noise estimate to subtract
extern "C" double noise_kind; // @noise_kind: 0=white 1=pink 2=brown 3=ou

// Number of selectable noise colors and their display names.
constexpr int NUM_KINDS = 4;
static const char *kKindNames[NUM_KINDS] = {"white", "pink", "brown", "ou"};

// Render one batch by invoking the kernel, which reads the current globals.
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

// How often the render thread rebuilds the batch. Bounds how quickly a wet /
// noise_kind change becomes audible.
constexpr auto REGEN_PERIOD = std::chrono::milliseconds(100);

// Rebuild the back buffer from the kernel and publish it. Only the render
// thread calls this once audio is live, so it owns gBackBuffer.
static void regenerate() {
    double *back = gBackBuffer;
    fillBatch(back, BATCH_SAMPLES);
    gBackBuffer = gActive.exchange(back, std::memory_order_acq_rel);
}

// Background render thread: periodically regenerate so global changes written
// from the keyboard thread are picked up asynchronously.
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

// Reprint the single status line (wet % and current noise color).
static void printStatus() {
    int kind = static_cast<int>(noise_kind);
    if (kind < 0) kind = 0;
    if (kind >= NUM_KINDS) kind = NUM_KINDS - 1;
    printf("\rNoise reduction: %3.0f%%   |   Noise color: %-6s   ",
           wet * 100.0, kKindNames[kind]);
    fflush(stdout);
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
    // and stay converged. Start fully wet (noise removed) with white noise.
    mu = 0.001;
    wet = 1.0;
    noise_kind = 0.0; // white
    regenerate();

    if (AudioOutputUnitStart(outputUnit) != noErr) {
        std::cerr << "Could not start audio stream output pipeline." << std::endl;
        return 1;
    }

    std::thread renderThread(renderLoop);

    std::cout << "\n==============================================" << std::endl;
    std::cout << "   CoreAudio LMS Adaptive Noise Canceller Running!" << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << " -> A 440 Hz tone corrupted by interference; the LMS filter" << std::endl;
    std::cout << "    estimates the interference and the knobs sweep how much" << std::endl;
    std::cout << "    of it to remove and which color of noise to fight." << std::endl;
    std::cout << " -> [UP Arrow]    add cancellation (toward a clean tone)" << std::endl;
    std::cout << " -> [DOWN Arrow]  back it off (bring the interference back)" << std::endl;
    std::cout << " -> [RIGHT Arrow] next noise color  (white -> pink -> brown -> ou)" << std::endl;
    std::cout << " -> [LEFT Arrow]  previous noise color" << std::endl;
    std::cout << " -> [Q] or Ctrl+C to stop the program safely" << std::endl;
    std::cout << "==============================================\n" << std::endl;

    enableRawMode();
    printStatus();

    char c;
    while (read(STDIN_FILENO, &c, 1) == 1 && c != 'q' && c != 'Q') {
        // Arrow keys send: '\x1b', '[', then 'A' Up / 'B' Down / 'C' Right / 'D' Left.
        if (c == '\x1b') {
            char seq[2];
            if (read(STDIN_FILENO, &seq[0], 1) == 1 && read(STDIN_FILENO, &seq[1], 1) == 1) {
                if (seq[0] == '[') {
                    if (seq[1] == 'A') {        // Up: more cancellation
                        wet = std::min(1.0, wet + 0.05);
                        printStatus();
                    } else if (seq[1] == 'B') { // Down: less cancellation
                        wet = std::max(0.0, wet - 0.05);
                        printStatus();
                    } else if (seq[1] == 'C') { // Right: next noise color
                        int k = (static_cast<int>(noise_kind) + 1) % NUM_KINDS;
                        noise_kind = static_cast<double>(k);
                        printStatus();
                    } else if (seq[1] == 'D') { // Left: previous noise color
                        int k = (static_cast<int>(noise_kind) + NUM_KINDS - 1) % NUM_KINDS;
                        noise_kind = static_cast<double>(k);
                        printStatus();
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
