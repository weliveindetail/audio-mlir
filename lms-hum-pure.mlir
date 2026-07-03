// Same 60 Hz hum canceller as lms-hum.mlir, but with NO dsp dialect: the signal
// generation and the LMS adaptive filter are spelled out directly in the core
// affine / arith / math / memref dialects. dsp1's dsp->affine lowering passes
// are no-ops here (there are no dsp ops), so this goes straight to LLVM.
//
// The point is an exact A/B: this must produce the *same* checksum as the
// built-in-op kernel, which proves the hand-written LMS == dsp.lmsFilterResponse.
// To stay bit-identical we reuse the exact constants dsp.getRangeOfVector emits
// -- it reads its first/step operands as 32-bit float, so 1/44100, 2*pi and 0.7
// come out slightly rounded; we hard-code those rounded values below.
module {
  memref.global "public" @mu : memref<f64> = dense<1.000000e-02>

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

    // t[i] = i*dt, built with the same running-sum recurrence the op emits
    // (i*dt would round differently -- the recurrence is what makes it match).
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

    // LMS adaptive FIR, 32 taps. w carries the feedback state across samples.
    %mumem = memref.get_global @mu : memref<f64>
    %muval = memref.load %mumem[] : memref<f64>
    %w = memref.alloc() : memref<32xf64>
    affine.for %i = 0 to 32 {
      affine.store %z, %w[%i] : memref<32xf64>
    }

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

    // cleaned output[n] = d[n] - y[n]
    affine.for %n = 0 to 44100 {
      %dn = affine.load %d[%n] : memref<44100xf64>
      %yn = affine.load %y[%n] : memref<44100xf64>
      %o  = arith.subf %dn, %yn : f64
      affine.store %o, %out[%n] : memref<44100xf64>
    }

    memref.dealloc %t : memref<44100xf64>
    memref.dealloc %x : memref<44100xf64>
    memref.dealloc %d : memref<44100xf64>
    memref.dealloc %y : memref<44100xf64>
    memref.dealloc %w : memref<32xf64>
    return
  }
}
