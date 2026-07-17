// Adaptive noise canceller (Widrow ANC) with a RUNTIME-selectable noise color
// driven by the @noise_kind knob via the dsp.index_switch op. dsp.index_switch
// is a first-class dsp op: a runtime index selector picks one of several regions
// and only the selected case runs.
//
//   x[n]  = noise color chosen at run time (white/pink/brown/ou, or none=silence)
//   n0[n] = 0.7 x[n] + 0.5 x[n-1] + 0.3 x[n-2]   (an "acoustic path", via delay)
//   d[n]  = sin(2*pi*440*t) + n0[n]              (tone buried in noise)
//   y     = LMS(x -> d), a 32-tap adaptive FIR   (dsp.lmsFilterResponse)
//   out   = d - wet*y                            (noise removed, tone revealed)
//
// dsp.index_switch mirrors scf.index_switch's syntax -- an index selector,
// integer `case` regions, a mandatory `default`, regions yielding a common
// result type (dsp.yield terminator). It is a distinct dsp op (not scf.index_-
// switch directly) so it travels through the tensor-land frontend and only
// bufferizes during affine lowering, where its own ConversionPattern rewrites it
// to a memref-yielding scf.index_switch (one function, a runtime branch). This
// is the intended, final lowering of dsp.index_switch.
//
// State model:
//   * ops INSIDE the switch regions (the noise generators) get INDEPENDENT
//     stream-state globals -- one per case, materialized by the stream-state
//     pass, which walks into the switch regions before lowering.
//   * ops OUTSIDE the switch (delay x2, LMS, tone phase) share ONE state global
//     each across all variants.
//   * the optional `reset = "region_local"` attribute is parsed but IGNORED by
//     this runtime-branch lowering; it is reserved for the planned, separate
//     dsp.variant_switch specialization/unswitch op (see samples/notes.md),
//     which would clone the enclosing @run per case, build a dispatcher, and
//     restart the noise state only on a switch.
//
// Cross-call persistence (block streaming) -- compile with --stream -- is
// unchanged from lms-noise.mlir; the @run signature is identical, so the
// existing interactive host drives this kernel as-is (Up/Down = @wet, and now
// Left/Right = @noise_kind).
module {
  // LMS adaptation rate (interactive, read at render time inside the kernel).
  memref.global "public" @mu : memref<f64> = dense<1.000000e-03>
  // Wet mix (interactive): fraction of the estimated noise to subtract.
  memref.global "public" @wet : memref<f64> = dense<1.000000e+00>
  // Noise color select (interactive): 0=white 1=pink 2=brown 3=ou, and any
  // other value ("none") = silence. Read once per @run call as the
  // dsp.index_switch selector; v1 assumes exactly this shape (a plain load of
  // a named global), mid-term an arbitrary SSA selector.
  memref.global "public" @noise_kind : memref<i64> = dense<0>

  dsp.func @run(%out: memref<44100xf64>) attributes {llvm.emit_c_interface} {
    // --- shared noise parameters (defined outside the switch; the regions
    //     reference them -- dsp.index_switch is NOT isolated-from-above) ---
    %seed  = dsp.constant dense<1.000000e+00> : tensor<f64>
    %sigma = dsp.constant dense<1.000000e+00> : tensor<f64>
    %n     = dsp.constant dense<4.410000e+04> : tensor<f64>

    // --- selector: load @noise_kind once, interpret as an index ---
    %kmem  = memref.get_global @noise_kind : memref<i64>
    %kval  = memref.load %kmem[] : memref<i64>
    %ksel  = arith.index_cast %kval : i64 to index

    // --- runtime noise-color reference x[n] ---
    // Cases 0..3 each hold a stateful noise generator with its OWN persistent
    // stream-state global; the default ("none") case is silence -- a stateless
    // zero vector (getRangeOfVector with step 0), so it carries no state global.
    // The reset="region_local" attribute (reserved for dsp.variant_switch) is
    // ignored here: under the runtime-branch lowering the noise streams simply
    // persist per case, so switching color resumes that case's stream.
    %x = dsp.index_switch %ksel {reset = "region_local"} -> tensor<44100xf64>
      case 0 {
        %w = "dsp.noise_white"(%n, %sigma, %seed) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<44100xf64>
        dsp.yield %w : tensor<44100xf64>
      }
      case 1 {
        %p = "dsp.noise_pink"(%n, %sigma, %seed) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<44100xf64>
        dsp.yield %p : tensor<44100xf64>
      }
      case 2 {
        %b = "dsp.noise_brown"(%n, %sigma, %seed) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<44100xf64>
        dsp.yield %b : tensor<44100xf64>
      }
      case 3 {
        %o = "dsp.noise_ou"(%n, %sigma, %seed) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<44100xf64>
        dsp.yield %o : tensor<44100xf64>
      }
      default {
        // "none": silence. Different signature from the noise ops (no
        // seed/sigma, no stream state) -- exactly why cases are regions, not an
        // op-rename. Zeros = getRangeOfVector(first=0, N, step=0).
        %z0 = dsp.constant dense<0.000000e+00> : tensor<f64>
        %s  = "dsp.getRangeOfVector"(%z0, %n, %z0) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<44100xf64>
        dsp.yield %s : tensor<44100xf64>
      }

    // --- acoustic path n0[n] = 0.7 x[n] + 0.5 x[n-1] + 0.3 x[n-2] ---
    // (unchanged from lms-noise.mlir; operates on the selected %x)
    %zero = dsp.constant dense<0.000000e+00> : tensor<f64>
    %cnt  = dsp.constant dense<4.410000e+04> : tensor<f64>
    %d1c  = dsp.constant dense<1.000000e+00> : tensor<f64>
    %d2c  = dsp.constant dense<2.000000e+00> : tensor<f64>
    %x1 = "dsp.delay"(%x, %d1c) : (tensor<44100xf64>, tensor<f64>) -> tensor<44100xf64>
    %x2 = "dsp.delay"(%x, %d2c) : (tensor<44100xf64>, tensor<f64>) -> tensor<44100xf64>

    %c07 = dsp.constant dense<7.000000e-01> : tensor<f64>
    %c05 = dsp.constant dense<5.000000e-01> : tensor<f64>
    %c03 = dsp.constant dense<3.000000e-01> : tensor<f64>
    %g07 = "dsp.getRangeOfVector"(%c07, %cnt, %zero) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<44100xf64>
    %g05 = "dsp.getRangeOfVector"(%c05, %cnt, %zero) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<44100xf64>
    %g03 = "dsp.getRangeOfVector"(%c03, %cnt, %zero) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<44100xf64>
    %p0 = "dsp.mul"(%x, %g07)  : (tensor<44100xf64>, tensor<44100xf64>) -> tensor<44100xf64>
    %p1 = "dsp.mul"(%x1, %g05) : (tensor<44100xf64>, tensor<44100xf64>) -> tensor<44100xf64>
    %p2 = "dsp.mul"(%x2, %g03) : (tensor<44100xf64>, tensor<44100xf64>) -> tensor<44100xf64>
    %n01 = "dsp.add"(%p0, %p1) : (tensor<44100xf64>, tensor<44100xf64>) -> tensor<44100xf64>
    %n0  = "dsp.add"(%n01, %p2) : (tensor<44100xf64>, tensor<44100xf64>) -> tensor<44100xf64>

    // --- 440 Hz tone: sin(2*pi*440 * t), t[n] = n*dt ---
    %dt    = dsp.constant dense<2.2675736961451248E-5> : tensor<f64>   // 1/44100
    %t     = "dsp.getRangeOfVector"(%zero, %cnt, %dt) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<44100xf64>
    %w440  = dsp.constant dense<2.764601535159018E+3> : tensor<f64>    // 2*pi*440
    %w440v = "dsp.getRangeOfVector"(%w440, %cnt, %zero) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<44100xf64>
    %arg   = "dsp.mul"(%t, %w440v) : (tensor<44100xf64>, tensor<44100xf64>) -> tensor<44100xf64>
    %tone  = dsp.sin(%arg : tensor<44100xf64>) to tensor<44100xf64>

    // --- desired d[n] = tone + colored noise ---
    %d = "dsp.add"(%tone, %n0) : (tensor<44100xf64>, tensor<44100xf64>) -> tensor<44100xf64>

    // --- 32-tap LMS: learn the noise path, weights persist across calls ---
    %mumem = memref.get_global @mu : memref<f64>
    %muval = memref.load %mumem[] : memref<f64>
    %mu    = tensor.from_elements %muval : tensor<f64>
    %flen  = dsp.constant dense<3.200000e+01> : tensor<f64>
    %y = "dsp.lmsFilterResponse"(%x, %d, %mu, %flen) : (tensor<44100xf64>, tensor<44100xf64>, tensor<f64>, tensor<f64>) -> tensor<44100xf64>

    // --- out = d - wet*y  (interactive wet mix; wet=1 -> full cancellation) ---
    %wetmem = memref.get_global @wet : memref<f64>
    %wetval = memref.load %wetmem[] : memref<f64>
    %wet    = tensor.from_elements %wetval : tensor<f64>
    %wy     = "dsp.gain"(%y, %wet) : (tensor<44100xf64>, tensor<f64>) -> tensor<44100xf64>
    %outt   = "dsp.sub"(%d, %wy) : (tensor<44100xf64>, tensor<44100xf64>) -> tensor<44100xf64>
    dsp.return %outt : tensor<44100xf64>
  }
}
