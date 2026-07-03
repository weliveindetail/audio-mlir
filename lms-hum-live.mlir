// Stateful variant of lms-hum-pure.mlir for the looping CoreAudio demo.
//
// The only difference from lms-hum-pure.mlir is that the 32 LMS weights live in
// a "public" global memref instead of a per-call alloc, so they PERSIST across
// _mlir_ciface_run invocations. The host re-renders the same one-second window
// every ~100 ms; with the weights carried over, the filter converges once at
// startup and every subsequent buffer is uniformly clean. That makes the looped
// buffer's head match its tail, removing the once-per-second click you get when
// the weights reset to zero each render.
//
// The weights are zero-initialized, so the FIRST render out of a fresh process
// is bit-identical to lms-hum-pure.mlir / lms-hum.mlir -- the A/B checksum
// (1.964609059359287e+01) still holds on the first buffer.
//
// Note: the reference-signal history (last 31 x samples) is still not carried
// across buffers, so a small boundary artifact in y[0..31] can remain.
module {
  memref.global "public" @mu : memref<f64> = dense<1.000000e-02>

  // Wet/dry mix for the hum estimate: out = d - wet*y. wet=0 leaves the full
  // 60 Hz hum in; wet=1 subtracts the whole converged estimate. This is the
  // interactive noise-reduction knob -- it scales only the output, never the
  // adaptation (the LMS keeps updating on the full error d-y), so y stays a
  // full-strength estimate regardless of where the knob sits.
  memref.global "public" @wet : memref<f64> = dense<1.000000e+00>

  // Persistent LMS weights (feedback state carried across renders).
  memref.global "public" @lms_weights : memref<32xf64> = dense<0.000000e+00>

  func.func @run(%out: memref<44100xf64>) attributes {llvm.emit_c_interface} {
    %z     = arith.constant 0.000000e+00 : f64
    %dt    = arith.constant 2.2675736545352265E-5 : f64   // float-rounded 1/44100
    %twopi = arith.constant 6.2831854820251465 : f64       // float-rounded 2*pi
    %f440  = arith.constant 4.400000e+02 : f64
    %f60   = arith.constant 6.000000e+01 : f64
    %hgain = arith.constant 0.69999998807907104 : f64      // float-rounded 0.7
    %c0    = arith.constant 0 : index

    %t = memref.alloc() : memref<44100xf64>   // time base (seconds)
    %x = memref.alloc() : memref<44100xf64>   // 60 Hz reference
    %d = memref.alloc() : memref<44100xf64>   // desired = tone + hum
    %y = memref.alloc() : memref<44100xf64>   // adaptive prediction of the hum

    // t[i] = i*dt, built with the same running-sum recurrence the op emits.
    affine.store %z, %t[%c0] : memref<44100xf64>
    %tlast = affine.for %i = 1 to 44100 iter_args(%acc = %z) -> (f64) {
      %v = arith.addf %acc, %dt : f64
      affine.store %v, %t[%i] : memref<44100xf64>
      affine.yield %v : f64
    }

    // x[i] = sin(2*pi*60*t)   ;   d[i] = sin(2*pi*440*t) + 0.7*x[i]
    affine.for %i = 0 to 44100 {
      %ti = affine.load %t[%i] : memref<44100xf64>
      %c440 = arith.mulf %ti, %f440 : f64
      %a440 = arith.mulf %c440, %twopi : f64
      %s440 = math.sin %a440 : f64
      %c60 = arith.mulf %ti, %f60 : f64
      %a60 = arith.mulf %c60, %twopi : f64
      %s60 = math.sin %a60 : f64
      affine.store %s60, %x[%i] : memref<44100xf64>
      %hum = arith.mulf %s60, %hgain : f64
      %di = arith.addf %s440, %hum : f64
      affine.store %di, %d[%i] : memref<44100xf64>
    }

    // LMS adaptive FIR, 32 taps. w is the PERSISTENT global -- no re-zeroing.
    %mumem = memref.get_global @mu : memref<f64>
    %muval = memref.load %mumem[] : memref<f64>
    %w = memref.get_global @lms_weights : memref<32xf64>

    affine.for %n = 0 to 44100 {
      affine.store %z, %y[%n] : memref<44100xf64>
      // y[n] = sum_i w[i] * x[n-i]
      affine.for %i = 0 to 32 {
        affine.if affine_set<(d0, d1) : (d0 - d1 >= 0)>(%n, %i) {
          %xni = affine.load %x[%n - %i] : memref<44100xf64>
          %wi  = affine.load %w[%i] : memref<32xf64>
          %p   = arith.mulf %xni, %wi : f64
          %yac = affine.load %y[%n] : memref<44100xf64>
          %yn  = arith.addf %p, %yac : f64
          affine.store %yn, %y[%n] : memref<44100xf64>
        }
      }
      // e[n] = d[n] - y[n]
      %dn = affine.load %d[%n] : memref<44100xf64>
      %yn = affine.load %y[%n] : memref<44100xf64>
      %e  = arith.subf %dn, %yn : f64
      // w[i] += mu * e[n] * x[n-i]
      affine.for %i = 0 to 32 {
        affine.if affine_set<(d0, d1) : (d0 - d1 >= 0)>(%n, %i) {
          %xni = affine.load %x[%n - %i] : memref<44100xf64>
          %wi  = affine.load %w[%i] : memref<32xf64>
          %ex  = arith.mulf %e, %xni : f64
          %mex = arith.mulf %muval, %ex : f64
          %wn  = arith.addf %wi, %mex : f64
          affine.store %wn, %w[%i] : memref<32xf64>
        }
      }
    }

    // cleaned output[n] = d[n] - wet*y[n]   (wet is the noise-reduction knob)
    %wetmem = memref.get_global @wet : memref<f64>
    %wetval = memref.load %wetmem[] : memref<f64>
    affine.for %n = 0 to 44100 {
      %dn = affine.load %d[%n] : memref<44100xf64>
      %yn = affine.load %y[%n] : memref<44100xf64>
      %wy = arith.mulf %wetval, %yn : f64
      %o  = arith.subf %dn, %wy : f64
      affine.store %o, %out[%n] : memref<44100xf64>
    }

    memref.dealloc %t : memref<44100xf64>
    memref.dealloc %x : memref<44100xf64>
    memref.dealloc %d : memref<44100xf64>
    memref.dealloc %y : memref<44100xf64>
    return
  }
}
