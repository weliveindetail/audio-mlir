#include <iostream>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <unistd.h>
#include <termios.h>
#include <cmath>
#include <AudioToolbox/AudioToolbox.h>
#include <CoreMIDI/CoreMIDI.h>
#include <CoreAudio/HostTime.h>

// Interactive CoreAudio host for the runtime-color-selectable LMS adaptive
// noise canceller (lms-noise.mlir), which uses dsp.index_switch on
// the @noise_kind selector to pick the noise generator at run time.
//
//   Up / Down arrows    -> @wet:        how much of the estimated noise to remove
//                                       (0 = noise back in, 1 = clean tone).
//   Left / Right arrows -> @noise_kind: cycle the noise color (white/pink/brown/
//                                       ou/none), the live dsp.index_switch case.
//   + / - keys          -> @lfo_period: the cutoff-sweep automation speed
//                                       (+ faster / - slower).
//
// The noise reference is generated in-kernel from an LCG stream (the same math
// the dsp.noise_* dialect ops encode); no RNG primitive or host-fed data is
// needed.

const double SAMPLE_RATE = 44100.0;

// The kernel writes exactly one block per @run call: memref<512xf64>. This must
// match the tensor/memref shape in lms-noise.mlir (N in its BLOCK SIZE note).
// There is no host-side loop any more: --stream state (noise/LMS/delay) and the
// in-kernel @sample_offset (tone phase) carry continuity across calls, so each
// block is the true next 512 samples of an endless stream.
const size_t BLOCK_SAMPLES = 128;

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

// @noise_kind: the dsp.index_switch selector (0=white 1=pink 2=brown 3=ou,
// any other value = silence). A discrete choice, so it is an integer global
// (memref<i64> in the kernel) rather than a float knob like @mu / @wet.
extern "C" int64_t noise_kind;

// @lfo_period: the cutoff-sweep LFO period in samples (the automation *speed*).
// Smaller = faster sweep. This parameter is no longer a directly-written global:
// the kernel now exposes a *timestamped setter*, @set_value_lfo_period(value,
// frame), and interpolates the change via its phase accumulator (see the
// lms-noise.mlir header). The host schedules a new value with a frame offset
// into the next block; here the +/- keys apply at frame 0 (start of next block).
// We keep a host-side shadow of the current value purely for the status display.
extern "C" void _mlir_ciface_set_value_lfo_period(int64_t value, int64_t frame);
static int64_t gLfoPeriod = 147000; // mirrors the kernel's @lfo_period default

// @lfo_mode: who computes the cutoff-sweep. 0 = KERNEL-side (the kernel's own
// phase-accumulator triangle), 1 = HOST-side (this program computes an arbitrary
// sweep SHAPE and streams it into the kernel as per-sample breakpoints). Toggled
// with the 'M' key. Mode B is the "heavy parameter traffic" path: each rendered
// block the host fires a burst of timestamped breakpoints through the setter
// below, and the kernel linearly interpolates them into the same alpha it would
// otherwise synthesize itself -- so an arbitrary shape, not just the triangle.
extern "C" int64_t lfo_mode;                    // the DSL's @lfo_mode global
extern "C" void _mlir_ciface_set_value_lfo_breakpoint(double value, int64_t frame);
static int64_t gLfoMode = 0;                    // host shadow of @lfo_mode (display)
static double  gHostPhase = 0.0;                // host LFO phase accumulator, [0,1)
// Breakpoints per block: sample the shape every LFO_BP_STRIDE frames (plus the
// block-end frame). 128/8 = 16 => ~16 setter calls per block, the heavy traffic
// this mode exists to exercise. The kernel linearly interpolates between them.
constexpr int LFO_BP_STRIDE = 8;

//===----------------------------------------------------------------------===//
// MIDI voices: OS MIDI events -> host voice allocation -> kernel (voice, frame)
//===----------------------------------------------------------------------===//
// The kernel exposes a fixed bank of NUM_VOICES sawtooth voices (see the
// @voice_* globals in lms-noise.mlir). A note is dispatched to it out-of-band
// through @set_note_event(voice, freqHz, gate, frame): the (value, frame) pair
// the notes.md "timestamped setter" direction calls for. The kernel steps that
// voice's gate at `frame` inside the next 128-sample block and one-pole-smooths
// it, so an event at an arbitrary instant is rendered click-free.
//
// Split of responsibilities:
//   * CoreMIDI thread  -> only parses note on/off and enqueues them (lock-free).
//   * render thread    -> drains the queue, does VOICE ALLOCATION (which note
//                         maps to which of the 8 hardware voices, stealing the
//                         oldest on overflow -- inherently sequential bookkeeping,
//                         a natural host job), and calls the kernel setter. Doing
//                         allocation + setter only on the render thread keeps the
//                         writes strictly ordered w.r.t. @run's consume, exactly
//                         like the LFO breakpoint setter (no cross-thread race).
constexpr int NUM_VOICES = 8;            // MUST match the kernel's memref<8x...>
extern "C" void _mlir_ciface_set_note_event(int64_t voice, double freq,
                                            double gate, int64_t frame);

// Lock-free SPSC queue of raw MIDI note events (CoreMIDI thread -> render thread).
struct MidiEvent {
    bool     on;       // true = note-on (vel>0), false = note-off
    uint8_t  note;     // MIDI note number 0..127
    uint64_t tNanos;   // event host time in nanoseconds (for frame placement)
};
constexpr size_t MIDI_Q_CAP = 1u << 10; // 1024 events
constexpr size_t MIDI_Q_MASK = MIDI_Q_CAP - 1;
static MidiEvent gMidiQ[MIDI_Q_CAP];
static std::atomic<size_t> gMidiHead{0}; // producer (CoreMIDI)
static std::atomic<size_t> gMidiTail{0}; // consumer (render thread)

// Host-side voice table (owned by the render thread).
struct HostVoice { int note; bool active; uint64_t order; };
static HostVoice gHostVoices[NUM_VOICES] = {};
static uint64_t gVoiceOrder = 1;         // monotonic "age" for oldest-voice steal
static std::atomic<int> gActiveVoices{0}; // for the status line only

static const double NS_PER_SAMPLE = 1e9 / SAMPLE_RATE;

// MIDI note number -> frequency in Hz (A4 = note 69 = 440 Hz).
static double noteToHz(int note) {
    return 440.0 * std::pow(2.0, (note - 69) / 12.0);
}

// CoreMIDI read callback. Runs on a CoreMIDI thread; do the minimum: parse
// note-on/off out of each packet and enqueue. No allocation, no kernel calls.
static void midiReadProc(const MIDIPacketList *pktlist, void *, void *) {
    const MIDIPacket *pkt = &pktlist->packet[0];
    for (UInt32 p = 0; p < pktlist->numPackets; ++p) {
        uint64_t tNanos = pkt->timeStamp ? AudioConvertHostTimeToNanos(pkt->timeStamp)
                                         : AudioConvertHostTimeToNanos(AudioGetCurrentHostTime());
        // Walk the (possibly multi-message) packet in 3-byte channel-voice chunks.
        for (UInt16 i = 0; i + 2 < pkt->length + 1 && i + 2 < pkt->length; i += 3) {
            uint8_t status = pkt->data[i] & 0xF0;
            uint8_t note   = pkt->data[i + 1] & 0x7F;
            uint8_t vel    = pkt->data[i + 2] & 0x7F;
            bool on;
            if (status == 0x90 && vel > 0)      on = true;
            else if (status == 0x80 || (status == 0x90 && vel == 0)) on = false;
            else continue; // ignore CC / pitchbend / aftertouch for this prototype
            size_t head = gMidiHead.load(std::memory_order_relaxed);
            size_t tail = gMidiTail.load(std::memory_order_acquire);
            if (head - tail < MIDI_Q_CAP) {     // drop on overflow rather than block
                gMidiQ[head & MIDI_Q_MASK] = MidiEvent{on, note, tNanos};
                gMidiHead.store(head + 1, std::memory_order_release);
            }
        }
        pkt = MIDIPacketNext(pkt);
    }
}

// --- computer-keyboard "emulated MIDI" (used when no real MIDI source) --------
// When setupMidi() finds no device we turn the computer keyboard into a one-octave
// piano. Terminal input gives no key-UP events, so the key acts as a TOGGLE: the
// first press enqueues a note-ON, the next press on the same key enqueues a
// note-OFF. The keyboard thread is still the sole producer on the SPSC queue.
// The OS auto-repeats a held key, which would flip the toggle rapidly, so we
// debounce: same-key presses within KBD_DEBOUNCE_NANOS of the last are ignored.
static bool gKbdPiano = false;
static bool gKbdNoteOn[128] = {};           // toggle state per note (input-thread owned)
static uint64_t gKbdLastPress[128] = {};    // last accepted press, ns (input-thread owned)
constexpr uint64_t KBD_DEBOUNCE_NANOS = 200'000'000ull; // ignore auto-repeat within 200 ms

// Home-row piano: a=C4(60) w s e d f t g y h u j k = up to C5(72).
static int keyToNote(char c) {
    switch (c) {
        case 'a': return 60; case 'w': return 61; case 's': return 62;
        case 'e': return 63; case 'd': return 64; case 'f': return 65;
        case 't': return 66; case 'g': return 67; case 'y': return 68;
        case 'h': return 69; case 'u': return 70; case 'j': return 71;
        case 'k': return 72;
        default:  return -1;
    }
}

// Release every voice currently playing `note` (render thread only).
static void releaseNote(int note, int64_t frame) {
    for (int i = 0; i < NUM_VOICES; ++i) {
        if (gHostVoices[i].active && gHostVoices[i].note == note) {
            gHostVoices[i].active = false;
            gActiveVoices.fetch_sub(1, std::memory_order_relaxed);
            _mlir_ciface_set_note_event(i, 0.0, 0.0, frame);
        }
    }
}

// Drain queued MIDI events, allocate voices, and dispatch (voice, freq, gate,
// frame) to the kernel. Called on the render thread right before rendering a
// block. `blockStartNanos` timestamps sample 0 of the block being rendered, used
// to place each event at its frame within [0,128).
static void drainMidiEvents(uint64_t blockStartNanos) {
    size_t tail = gMidiTail.load(std::memory_order_relaxed);
    size_t head = gMidiHead.load(std::memory_order_acquire);
    for (; tail != head; ++tail) {
        MidiEvent ev = gMidiQ[tail & MIDI_Q_MASK];
        int64_t frame = 0;
        if (ev.tNanos > blockStartNanos) {
            double f = (ev.tNanos - blockStartNanos) / NS_PER_SAMPLE;
            frame = f < 0 ? 0 : (f > 127 ? 127 : static_cast<int64_t>(f));
        }
        if (ev.on) {
            // Find a free voice, else steal the oldest active one.
            int v = -1;
            for (int i = 0; i < NUM_VOICES; ++i)
                if (!gHostVoices[i].active) { v = i; break; }
            if (v < 0) {
                v = 0;
                for (int i = 1; i < NUM_VOICES; ++i)
                    if (gHostVoices[i].order < gHostVoices[v].order) v = i;
            } else {
                gActiveVoices.fetch_add(1, std::memory_order_relaxed);
            }
            gHostVoices[v] = HostVoice{ev.note, true, gVoiceOrder++};
            _mlir_ciface_set_note_event(v, noteToHz(ev.note), 1.0, frame);
        } else {
            releaseNote(ev.note, frame);
        }
    }
    gMidiTail.store(tail, std::memory_order_release);
}

// Create a CoreMIDI client + input port and connect every MIDI source the OS
// currently sees (hardware keyboards, IAC virtual buses, etc.). Returns false if
// CoreMIDI is unavailable; the demo still runs (just silent until a note plays).
static bool setupMidi() {
    MIDIClientRef client = 0;
    if (MIDIClientCreate(CFSTR("lms-noise"), nullptr, nullptr, &client) != noErr)
        return false;
    MIDIPortRef inPort = 0;
    if (MIDIInputPortCreate(client, CFSTR("in"), midiReadProc, nullptr, &inPort) != noErr)
        return false;
    ItemCount nSrc = MIDIGetNumberOfSources();
    for (ItemCount i = 0; i < nSrc; ++i)
        MIDIPortConnectSource(inPort, MIDIGetSource(i), nullptr);
    return nSrc > 0;
}

// Automation-speed bounds (samples at 44100): ~10 Hz (fast) to ~0.05 Hz (~20 s).
constexpr int64_t LFO_PERIOD_MIN = 4410;   // 44100 / 10  -> 10 Hz
constexpr int64_t LFO_PERIOD_MAX = 882000; // 44100 * 20  -> 0.05 Hz
constexpr double LFO_STEP = 1.25; // multiplicative step per key press

// Selectable noise colors and their display names. Index 4 ("none") drives the
// index_switch default (silence), so the knob sweeps every region.
constexpr int NUM_KINDS = 5;
static const char *kKindNames[NUM_KINDS] = {"white", "pink", "brown", "ou",
                                            "none"};

// Render one block by invoking the kernel, which reads the current globals and
// advances its own --stream / @sample_offset state.
static void fillBlock(double *buf) {
    MemRefDescriptor1D desc{buf, buf, 0, static_cast<int64_t>(BLOCK_SAMPLES), 1};
    _mlir_ciface_run(&desc);
}

static std::atomic<bool> gRunning{true};

//===----------------------------------------------------------------------===//
// Lock-free SPSC ring buffer (render thread -> audio callback)
//===----------------------------------------------------------------------===//
// The kernel still does ~16 malloc/free per @run call, so it is NOT real-time
// safe to invoke inside the CoreAudio callback. Instead a background render
// thread produces 512-sample blocks into this ring and the callback only copies
// out -- no allocation, no locks on the audio thread. The ring decouples the
// producer's block granularity from the callback's frame count.
//
// Capacity is a power of two so index wrap is a mask. Single producer / single
// consumer: the producer owns gHead, the consumer owns gTail; each publishes its
// index with release and reads the other's with acquire.
constexpr size_t RING_CAPACITY = 1u << 13; // 8192 samples (~16 blocks, ~186 ms)
constexpr size_t RING_MASK = RING_CAPACITY - 1;
static double gRing[RING_CAPACITY];
static std::atomic<size_t> gHead{0}; // next write index (producer)
static std::atomic<size_t> gTail{0}; // next read index  (consumer)

// Background render thread: keep the ring as full as possible, one kernel block
// at a time, so wet / noise_kind changes written from the keyboard thread are
// picked up within a block or two. Renders into a scratch block then copies into
// the ring (the kernel needs a contiguous 512-wide memref).
static void emitLfoBlock(); // defined below (with the LFO helpers)

static void renderLoop() {
    double scratch[BLOCK_SAMPLES];
    while (gRunning.load(std::memory_order_relaxed)) {
        size_t head = gHead.load(std::memory_order_relaxed);
        size_t tail = gTail.load(std::memory_order_acquire);
        size_t used = head - tail;
        if (used + BLOCK_SAMPLES > RING_CAPACITY) {
            // Ring is full enough; wait for the callback to drain a bit.
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }
        // Drain MIDI first: allocate voices and stage this block's note events
        // (voice, freq, gate, frame) into the kernel, timestamped against now.
        uint64_t nowNanos = AudioConvertHostTimeToNanos(AudioGetCurrentHostTime());
        drainMidiEvents(nowNanos);
        emitLfoBlock();  // Mode B: stream this block's LFO shape as breakpoints
        fillBlock(scratch);
        for (size_t i = 0; i < BLOCK_SAMPLES; ++i)
            gRing[(head + i) & RING_MASK] = scratch[i];
        gHead.store(head + BLOCK_SAMPLES, std::memory_order_release);
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

// Map the current @noise_kind to a display name. A value outside 0..3 selects
// the index_switch default ("none"/silence).
static const char *currentKindName() {
    int k = static_cast<int>(noise_kind);
    if (k < 0 || k >= NUM_KINDS - 1)
        return kKindNames[NUM_KINDS - 1]; // "none"
    return kKindNames[k];
}

// Current cutoff-sweep rate in Hz (LFO frequency = sample rate / period).
static double lfoHz() {
    return SAMPLE_RATE / static_cast<double>(gLfoPeriod);
}

// Scale the automation speed by `factor` (>1 = faster sweep -> shorter period),
// clamped to the sensible band. Rounds so the period stays a whole number.
// Instead of writing the @lfo_period global directly, schedule the change via
// the kernel's timestamped setter at frame 0 (apply from the start of the next
// rendered block); the kernel's phase accumulator interpolates it click-free.
static void scaleLfoSpeed(double factor) {
    double p = static_cast<double>(gLfoPeriod) / factor; // faster => smaller period
    int64_t np = static_cast<int64_t>(p + 0.5);
    if (np < LFO_PERIOD_MIN) np = LFO_PERIOD_MIN;
    if (np > LFO_PERIOD_MAX) np = LFO_PERIOD_MAX;
    gLfoPeriod = np;
    _mlir_ciface_set_value_lfo_period(np, /*frame=*/0);
}

// The host-side LFO shape as a function of phase in [0,1). Default: the SAME
// triangle the kernel synthesizes in Mode A -- alpha = 0.02 + 0.33*(1-|2f-1|) --
// so the two modes sound alike (a clean A/B check). Swap this body for any
// function to get an arbitrary cutoff-sweep shape, which is the whole point of
// the host-side mode (the kernel triangle can only ever be a triangle).
static double lfoShape(double phase) {
    double frac = phase - std::floor(phase);
    double tri = 1.0 - std::fabs(2.0 * frac - 1.0);
    return 0.02 + 0.33 * tri;
}

// Per-block host LFO update, called on the render thread right before the kernel
// invocation. Always advances the host phase (so toggling modes stays roughly
// continuous); only STREAMS breakpoints when Mode B is active. Because it runs
// immediately before _mlir_ciface_run on the same thread, the breakpoint writes
// and the kernel's consume/reset of @lfo_bp are strictly sequential -- no
// cross-thread race on the array.
static void emitLfoBlock() {
    double inc = 1.0 / static_cast<double>(gLfoPeriod); // cycles per sample
    if (lfo_mode != 0) {
        const int N = static_cast<int>(BLOCK_SAMPLES);
        for (int f = 0; f < N; f += LFO_BP_STRIDE)
            _mlir_ciface_set_value_lfo_breakpoint(lfoShape(gHostPhase + f * inc), f);
        // Anchor the final frame so the last segment ramps to the right value
        // instead of the interpolator holding the previous breakpoint flat.
        _mlir_ciface_set_value_lfo_breakpoint(lfoShape(gHostPhase + (N - 1) * inc), N - 1);
    }
    gHostPhase += BLOCK_SAMPLES * inc;
    gHostPhase -= std::floor(gHostPhase); // keep in [0,1)
}

// Compact status readout: captions on one line, values redrawn on the line below
// (so it fits narrow terminals and stays easy to refresh -- captions are static
// and printed once by printCaptions(); printStatus() only rewrites the value row).
// Both rows share the same column widths so cells line up.
#define STATUS_FMT "%-8s| %-7s| %-9s| %-7s| %-6s"

static void printCaptions() {
    printf(STATUS_FMT "\n", "Reduce", "Color", "Sweep", "LFO", "Voices");
    fflush(stdout);
}

static void printStatus() {
    char reduce[16], color[16], sweep[16], lfo[16], voices[16];
    snprintf(reduce, sizeof reduce, "%.0f%%", wet * 100.0);
    snprintf(color,  sizeof color,  "%s", currentKindName());
    snprintf(sweep,  sizeof sweep,  "%.2f Hz", lfoHz());
    snprintf(lfo,    sizeof lfo,    "%s", lfo_mode ? "HOST" : "KERNEL");
    snprintf(voices, sizeof voices, "%d/%d",
             gActiveVoices.load(std::memory_order_relaxed), NUM_VOICES);
    printf("\r" STATUS_FMT, reduce, color, sweep, lfo, voices);
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

    size_t tail = gTail.load(std::memory_order_relaxed);
    size_t head = gHead.load(std::memory_order_acquire);
    size_t avail = head - tail;

    for (UInt32 i = 0; i < inNumberFrames; ++i) {
        float s = 0.0f; // underrun -> silence rather than a stale/looped sample
        if (i < avail) {
            s = static_cast<float>(gRing[tail & RING_MASK]); // f64 -> f32 on copy
            ++tail;
        }
        leftChannel[i] = s;
        rightChannel[i] = s;
    }

    gTail.store(tail, std::memory_order_release);
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

    // Pin the callback frame count to one kernel block (N=512). This is a hint:
    // the ring buffer already decouples producer blocks from callback frames, so
    // playback is correct even if CoreAudio renegotiates a different slice size.
    // (a) MaximumFramesPerSlice bounds the frames the unit will ever ask for.
    UInt32 maxFrames = BLOCK_SAMPLES;
    AudioUnitSetProperty(outputUnit,
                         kAudioUnitProperty_MaximumFramesPerSlice,
                         kAudioUnitScope_Global,
                         0,
                         &maxFrames,
                         sizeof(maxFrames));
    // (b) Ask the default output *device* to use a 512-frame hardware buffer so
    //     the callback is actually invoked once per block.
    AudioDeviceID outputDevice = 0;
    UInt32 devSize = sizeof(outputDevice);
    AudioObjectPropertyAddress devAddr = {
        kAudioHardwarePropertyDefaultOutputDevice,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain};
    if (AudioObjectGetPropertyData(kAudioObjectSystemObject, &devAddr, 0, nullptr,
                                   &devSize, &outputDevice) == noErr) {
        UInt32 frameSize = BLOCK_SAMPLES;
        AudioObjectPropertyAddress bufAddr = {
            kAudioDevicePropertyBufferFrameSize,
            kAudioObjectPropertyScopeOutput,
            kAudioObjectPropertyElementMain};
        AudioObjectSetPropertyData(outputDevice, &bufAddr, 0, nullptr,
                                   sizeof(frameSize), &frameSize);
    }

    if (AudioUnitInitialize(outputUnit) != noErr) {
        std::cerr << "Could not initialize Core Audio hardware context buffers." << std::endl;
        return 1;
    }

    // Fix mu small so the persistent LMS weights converge over the first few
    // blocks and stay converged. Start fully wet (noise removed) with white
    // noise. The kernel's --stream state (and @sample_offset) persist across the
    // 512-sample @run calls, so there is no host-side loop.
    mu = 0.001;
    wet = 1.0;
    noise_kind = 0; // start on white

    // Connect to the OS MIDI graph. The kernel's tone bank is silent until a
    // note-on arrives, so play a MIDI keyboard (or route an IAC/virtual source).
    // With no device, fall back to the computer keyboard as a one-octave piano.
    bool midiOk = setupMidi();
    gKbdPiano = !midiOk;

    // Prime the ring with a few blocks so the very first callbacks never
    // underrun before the render thread has spun up.
    for (int i = 0; i < 4; ++i) {
        double scratch[BLOCK_SAMPLES];
        fillBlock(scratch);
        size_t head = gHead.load(std::memory_order_relaxed);
        for (size_t j = 0; j < BLOCK_SAMPLES; ++j)
            gRing[(head + j) & RING_MASK] = scratch[j];
        gHead.store(head + BLOCK_SAMPLES, std::memory_order_release);
    }

    if (AudioOutputUnitStart(outputUnit) != noErr) {
        std::cerr << "Could not start audio stream output pipeline." << std::endl;
        return 1;
    }

    std::thread renderThread(renderLoop);

    std::cout << "\n==============================================" << std::endl;
    std::cout << "   CoreAudio LMS Adaptive Noise Canceller Running!" << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << " -> Play a MIDI keyboard: each note triggers a polyphonic (max "
              << NUM_VOICES << ")" << std::endl;
    std::cout << "    sawtooth voice, buried in noise; the LMS filter estimates the" << std::endl;
    std::cout << "    interference and the knobs sweep how much of it to remove." << std::endl;
    std::cout << (midiOk ? " -> MIDI: connected to OS source(s)."
                         : " -> MIDI: no source found -- computer keyboard emulates one.")
              << std::endl;
    if (gKbdPiano)
        std::cout << " -> [A W S E D F T G Y H U J K] one-octave piano (C4..C5) -- press to start, press again to stop" << std::endl;
    std::cout << " -> [UP/DOWN]     more / less cancellation (toward a clean tone)" << std::endl;
    std::cout << " -> [LEFT/RIGHT]  cycle noise color (white/pink/brown/ou/none)" << std::endl;
    std::cout << " -> [+ / -]       cutoff-sweep speed (faster / slower)" << std::endl;
    std::cout << " -> [M]           LFO source: KERNEL triangle vs HOST breakpoints" << std::endl;
    std::cout << " -> [Q] or Ctrl+C to stop the program safely" << std::endl;
    std::cout << "==============================================\n" << std::endl;

    enableRawMode();
    printCaptions();
    printStatus();

    char c;
    while (read(STDIN_FILENO, &c, 1) == 1 && c != 'q' && c != 'Q') {
        // +/- (and =/_) change the cutoff-sweep automation speed. '+' speeds it
        // up (shorter LFO period), '-' slows it down.
        if (c == '+' || c == '=') {
            scaleLfoSpeed(LFO_STEP);
            printStatus();
            continue;
        }
        if (c == '-' || c == '_') {
            scaleLfoSpeed(1.0 / LFO_STEP);
            printStatus();
            continue;
        }
        // 'M' toggles who computes the cutoff-sweep LFO: the kernel's own triangle
        // (Mode A) or the host streaming an arbitrary shape as breakpoints (B).
        if (c == 'm' || c == 'M') {
            gLfoMode ^= 1;
            lfo_mode = gLfoMode;
            printStatus();
            continue;
        }
        // Computer-keyboard piano (only when no real MIDI device): a note key
        // TOGGLES its voice -- first press enqueues a note-ON, next press a
        // note-OFF. A per-key debounce drops the OS auto-repeat of a held key so
        // it doesn't flip the toggle. This is the sole producer on the MIDI queue.
        if (gKbdPiano) {
            int note = keyToNote(c);
            if (note >= 0) {
                uint64_t t = AudioConvertHostTimeToNanos(AudioGetCurrentHostTime());
                if (t - gKbdLastPress[note] >= KBD_DEBOUNCE_NANOS) {
                    gKbdLastPress[note] = t;
                    bool on = !gKbdNoteOn[note];
                    gKbdNoteOn[note] = on;
                    size_t head = gMidiHead.load(std::memory_order_relaxed);
                    size_t tail = gMidiTail.load(std::memory_order_acquire);
                    if (head - tail < MIDI_Q_CAP) {
                        gMidiQ[head & MIDI_Q_MASK] =
                            MidiEvent{on, static_cast<uint8_t>(note), t};
                        gMidiHead.store(head + 1, std::memory_order_release);
                    }
                }
                continue;
            }
        }
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
                    }
                    // Left/Right cycle the @noise_kind index_switch selector.
                    else if (seq[1] == 'C') { // Right: next color
                        noise_kind = (noise_kind + 1) % NUM_KINDS;
                        printStatus();
                    } else if (seq[1] == 'D') { // Left: prev color
                        noise_kind = (noise_kind + NUM_KINDS - 1) % NUM_KINDS;
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
