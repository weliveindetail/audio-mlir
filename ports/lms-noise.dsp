// =============================================================================
//  lms-noise.dsp -- a Faust port of the DSP-MLIR `samples/lms-noise.mlir` kernel
//  (the current polyphonic, MIDI-triggered revision).
//
//      x[n]   = colored noise (white/pink/brown/ou, or none = silence)
//      n0[n]  = 0.7 x[n] + 0.5 x[n-1] + 0.3 x[n-2]      (a 3-tap "acoustic path")
//      saw_v  = 2*frac(f_v t) - 1                        (per MIDI-triggered voice)
//      tone_v = lowPass(saw_v, alpha_v(t))              (per-voice swept 1-pole IIR)
//      tone   = 0.2 * sum_v gate_v * tone_v             (mixed voice bank)
//      d[n]   = tone[n] + n0[n]                          (tone buried in noise)
//      yhat   = LMS(x -> d), 32-tap adaptive FIR         (estimates the noise)
//      out    = d - wet*yhat                             (noise removed, tone revealed)
//
//  How this maps onto the kernel
//  -----------------------------
//  The MLIR kernel is block-based (one run(out) per 128-sample block, state in
//  module globals). Faust is natively per-sample and functional, so every
//  recurrence becomes a `~` feedback and the block bookkeeping disappears.
//
//    * The old continuous 440 Hz tone is gone: the kernel renders a bank of 8
//      sawtooth VOICES, silent until a MIDI note triggers one. Here that is the
//      standard Faust polyphonic instrument: `process` is ONE voice driven by the
//      host `freq`/`gain`/`gate` endpoints, and [nvoices:8] instantiates the bank
//      + voice allocation (the kernel's 8-slot decode loop).
//    * Each voice runs its OWN trigger-anchored cutoff LFO: at note-on the kernel
//      resets @voice_cut_phase to 0 and LATCHES @cut_lfo_step, then advances the
//      phase per sample and reads a SHARED @voice_cut_shape table (one cycle of a
//      looping wah). We mirror that: `cutShape` is one shared rdtable (the native
//      driver's default triangle wah -- the shape a host GUI would otherwise
//      draw), each voice reads it at its own reset-anchored phase, and the sweep
//      speed is sample-and-held at note-on so a live change only affects new notes.
//    * The per-voice one-pole low-pass and the smoothed gate reset at note-on,
//      exactly like the kernel's coeff-masked scans (a stolen voice starts a fresh
//      filter and attack, not the previous note's tail).
//    * n0 (acoustic path), the runtime noise-color switch, the 32-tap LMS and the
//      wet mix live in the mono `effect`, which the summed voices feed into --
//      the block pipeline's d = tone + n0, y = LMS(x->d), out = d - wet*y.
//
//  Build / run (polyphonic + MIDI), e.g.:
//      faust2caqt -midi -nvoices 8 lms-noise.dsp
//      faust2jaqt -midi -nvoices 8 lms-noise.dsp
// =============================================================================

import("stdfaust.lib");

declare name        "LMSNoise";
declare description  "Adaptive (Widrow LMS) noise canceller -- Faust port of the DSP-MLIR lms-noise kernel";
declare options     "[midi:on][nvoices:8]";

// ---- interactive knobs (were `memref.global "public"` in the MLIR kernel) ----
wet        = hslider("Noise Cancel", 0, 0, 1, 0.01) : si.smoo;   // wet mix
mu         = hslider("LMS Rate [scale:log]", 0.001, 0.00001, 0.01, 0.00001) : si.smoo;
sweepRate  = hslider("Sweep Rate [unit:Hz]", 3, 0.05, 10, 0.01); // cutoff-LFO speed
noiseColor = int(hslider("Noise Color", 0, 0, 4, 1));            // 0..4 selector

// ---- host-driven per-voice endpoints (the [nvoices] architecture writes these) -
freq = hslider("freq [unit:Hz]", 440, 20, 20000, 0.01);
gain = hslider("gain", 1, 0, 1, 0.01);
gate = button("gate");

// =============================================================================
//  Shared cutoff-SHAPE table (== @voice_cut_shape, memref<8000xf64>). One cycle
//  of the looping wah: alpha(t) = min + span*|2t-1|, t = i/L -- bright (0.35) at
//  the wrap (t=0 and t->1), muffled (0.02) at mid-cycle, periodic at the ends so
//  the wrap is click-free. Every voice reads THIS one table at its own phase.
// =============================================================================
L        = 8000;                      // MUST match the kernel memref<8000xf64>
alphaMin = 0.02;                      // muffled floor (dip of the wah)
alphaSpan = 0.33;                     // bright(0.35) - muffled(0.02)

cutTable   = alphaMin + alphaSpan * abs(2.0 * float(ba.time) / float(L) - 1.0);
cutAlpha(ph) = rdtable(L, cutTable, idx)
with {
    idx = min(L - 1, int(ph * float(L)));   // ph in [0,1) -> table index 0..L-1
};

// A phasor in [0,1) that HARD-RESETS to 0 whenever `reset` fires (the kernel's
// note-on phase restart). Used for both the saw and the cutoff LFO.
rphasor(inc, reset) = loop ~ _
with {
    loop(p) = ma.frac((p + inc) * (1.0 - reset));
};

// One-pole low-pass y[n] = (1-alpha)(1-reset) y[n-1] + alpha x[n]: time-varying
// cutoff, state cleared at note-on (== the kernel's aLp = (1-alpha)*(1-trg)).
onepole(alpha, reset, x) = loop ~ _
with {
    loop(y) = (y * (1.0 - alpha)) * (1.0 - reset) + alpha * x;
};

// =============================================================================
//  ONE VOICE (the [nvoices] bank replicates this). A MIDI-triggered sawtooth
//  through its own trigger-anchored swept low-pass. `process` == the kernel's
//  per-voice lane of the rank-2 tensor<8x128> bank.
// =============================================================================
voice = 0.2 * gain * gateSmooth * tone
with {
    // note-on == rising edge of the gate (the kernel's eventToTrigger pulse).
    trig = gate > gate';

    // Latch the sweep speed at note-on (== @voice_cut_step := @cut_lfo_step): a
    // live Sweep change then only re-speeds notes played afterwards.
    sweepHz = sweepRate : ba.sAndH(trig);

    // sawtooth: 2*frac(f t) - 1, phase restarted at note-on.
    saw = 2.0 * rphasor(freq / ma.SR, trig) - 1.0;

    // per-voice cutoff LFO phase in [0,1), restarted + re-sped at note-on; alpha
    // read from the shared shape at THIS voice's phase.
    cutPh = rphasor(sweepHz / ma.SR, trig);
    alpha = cutAlpha(cutPh);
    tone  = onepole(alpha, trig, saw);

    // one-pole gate smoother g += 0.01*(gate - g) for a click-free attack/release
    // (== the kernel's akc = 0.01 gate scan).
    gateSmooth = gate : *(0.01) : + ~ *(0.99);
};

process = voice;

// =============================================================================
//  MONO EFFECT: bury the mixed voice tone in colored noise, then learn + subtract
//  the noise path. Applied ONCE to the summed voices == the kernel's post-reduce
//  pipeline `d = tone + n0`, `y = LMS(x->d)`, `out = d - wet*y`.
// =============================================================================

// Runtime noise-color select (== the kernel's dsp.index_switch on @noise_kind).
// This is the true analog of the kernel's dsp.index_switch: only the SELECTED
// case runs, and the unselected cases FREEZE their internal state. `ba.selectn`
// would instead be strict -- computing all 5 generators every sample and picking
// one -- so the idle noise streams would keep advancing (audibly identical, but
// wasteful and NOT the kernel's semantics). `enable(sig, cond)` is Faust's
// signal-control primitive: it computes `sig` only while `cond` holds and holds
// its state otherwise, so a disabled generator emits 0 and its recurrences pause.
// Summing the five gated branches yields exactly the one active generator, and
// switching color RESUMES that case's stream from where it was frozen -- the same
// "switching color resumes that case's stream" behaviour the kernel's runtime-
// branch index_switch has (the reserved reset-on-switch variant is a separate op).
// Amplitudes are matched to roughly bipolar [-1,1].
gen(0) = no.noise;                                     // white
gen(1) = no.pink_noise * 12.5;                         // pink (pinking filter, loudness-matched)
gen(2) = no.noise : (+ ~ *(0.998)) : *(0.998 / 32.0) : max(-1.0) : min(1.0);  // brown
gen(3) = no.noise : *(0.1) : + ~ *(0.95);              // ou[n] = 0.95 ou[n-1] + 0.1 w
gen(4) = 0.0;                                          // "none" = silence
noiseSel = sum(k, 5, enable(gen(k), noiseColor == k));

// 32-tap LMS adaptive FIR. Returns the error e = d - yhat (== the cleaned signal
// at wet=1). The tap weights are per-tap integrators of mu*e*x[n-k]; the error is
// closed back through a single `~` so the recurrence has a well-defined one-sample
// delay (Faust rejects the undelayed mutual recursion).
numTaps = 32;
lms(x, d) = eLoop ~ _
with {
    eLoop(ePrev) = d - yhat
    with {
        w(k)  = (mu * ePrev * (x@k)) : + ~ _;          // adaptive tap weight
        yhat  = sum(k, numTaps, w(k) * (x@k));
    };
};

// Mixed tone in, cleaned signal out. yhat = d - e, so
//   out = d - wet*yhat = (1-wet)*d + wet*e   (wet=1 -> full cancellation).
anc(tone) = (1.0 - wet) * d + wet * e
with {
    x  = noiseSel;
    n0 = 0.7 * x + 0.5 * (x@1) + 0.3 * (x@2);           // 3-tap acoustic path
    d  = tone + n0;
    e  = lms(x, d);
};

effect = anc;
